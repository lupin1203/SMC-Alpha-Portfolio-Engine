# regime.py
"""
市場環境 (Regime) 判斷：回測引擎與即時雷達共用同一套規則，避免兩邊各說各話。

規則：
- 趨勢：0050 收盤 > 200 日均線 → Bull，否則 Bear
- 波動：0050 近 20 日年化波動率 > 25% → HighVol，否則 LowVol
"""
import numpy as np
import pandas as pd

ALL_REGIMES = ["Bull_LowVol", "Bull_HighVol", "Bear_LowVol", "Bear_HighVol"]

# README 中經驗證為正期望值的進場環境
DEFAULT_ALLOWED_REGIMES = ("Bull_LowVol", "Bear_HighVol")

TREND_SMA_WINDOW = 200
VOL_WINDOW = 20
HIGH_VOL_THRESHOLD = 25.0  # 年化波動率 (%)


def add_regime_columns(market_df):
    """在大盤 DataFrame 上加入 SMA200、Volatility_20d、Regime 欄位。"""
    if market_df is None or market_df.empty:
        return market_df
    market_df = market_df.copy()
    market_df['SMA200'] = market_df['Close'].rolling(TREND_SMA_WINDOW).mean()
    daily_ret = market_df['Close'].pct_change()
    market_df['Volatility_20d'] = daily_ret.rolling(VOL_WINDOW).std() * np.sqrt(252) * 100

    trend = np.where(market_df['Close'] > market_df['SMA200'], 'Bull', 'Bear')
    vol = np.where(market_df['Volatility_20d'] > HIGH_VOL_THRESHOLD, 'HighVol', 'LowVol')
    regime = pd.Series([f"{t}_{v}" for t, v in zip(trend, vol)], index=market_df.index)
    # 資料不足 (均線或波動率尚未成形) 時標記為 Unknown
    regime[market_df['SMA200'].isna() | market_df['Volatility_20d'].isna()] = 'Unknown'
    market_df['Regime'] = regime
    return market_df


def regime_at(market_df, date):
    """取得指定日期 (含之前最近一個交易日) 的市場環境。market_df 需先經過 add_regime_columns。"""
    if market_df is None or market_df.empty or 'Regime' not in market_df.columns:
        return 'Unknown'
    idx = market_df.index.get_indexer([pd.Timestamp(date)], method='pad')[0]
    if idx < 0:
        return 'Unknown'
    return market_df['Regime'].iloc[idx]
