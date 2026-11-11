# research_pipeline.py
"""
單一股票的特徵消融實驗 (Ablation Study)：把各個濾網逐一開關，看哪個真的有貢獻。

用法：
    from data_provider import fetch_stock_data
    from research_pipeline import QuantResearchPipeline
    df = fetch_stock_data("2330.TW", days=3650)
    report = QuantResearchPipeline(df).run_ablation_study({'min_gap_pct': 0.3})
    print(report.to_markdown(index=False))

只看研究期的話，先自己用 df[df.index < config.HOLDOUT_START] 截斷再丟進來。
"""
import itertools

import pandas as pd

from backtest_engine import run_backtest_core, summarize_trades
from regime import add_regime_columns, regime_at
from strategy import SMCStrategy

# 參數名必須和 strategy.SMCStrategy.generate_signals 實際認得的名稱一致，
# 寫錯的話策略會靜靜忽略，消融實驗每一列結果都一樣。
ABLATION_FEATURES = ['use_sma_filter', 'use_bos_filter', 'use_vol_filter', 'use_atr_filter']


def default_backtest(signal_df, bt_mode="移動停利", bt_val=10.0, realistic=True):
    """預設回測：實盤真實模式，回傳 (統計 dict, 交易明細)。"""
    trades, _ = run_backtest_core(signal_df, bt_mode=bt_mode, bt_val=bt_val, realistic=realistic)
    return summarize_trades(trades), trades


class QuantResearchPipeline:
    def __init__(self, df, strategy_class=SMCStrategy, backtest_engine=default_backtest):
        """
        :param df: 原始 OHLCV 數據
        :param strategy_class: BaseStrategy 的實作 (預設 SMCStrategy)
        :param backtest_engine: 函數，吃訊號 DataFrame，回傳 (統計 dict, 交易明細 DataFrame)
        """
        self.df = df
        self.strategy = strategy_class()
        self.backtest_engine = backtest_engine

    def run_ablation_study(self, base_params=None, features=ABLATION_FEATURES):
        """把 features 裡的濾網全部開關組合跑一遍，依期望值排序。"""
        base_params = dict(base_params or {})
        results = []
        for combo in itertools.product([True, False], repeat=len(features)):
            switches = dict(zip(features, combo))
            signal_df = self.strategy.generate_signals(self.df.copy(), **{**base_params, **switches})
            stats, _ = self.backtest_engine(signal_df)
            results.append({
                **switches,
                '訊號數': int(signal_df['FVG'].sum()),
                '交易次數': stats['trades'],
                '勝率(%)': round(stats['win_rate'] * 100, 1),
                '期望值(R)': round(stats['expectancy'], 2),
                '總淨利(R)': round(stats['total_r'], 2),
                '最大回撤(R)': round(stats['max_dd_r'], 2),
            })
        return pd.DataFrame(results).sort_values(by='期望值(R)', ascending=False).reset_index(drop=True)

    def run_regime_analysis(self, trades_df, market_df):
        """
        依進場當天的市場環境 (regime.py 的規則，和回測、雷達同一套) 統計交易表現。
        :param trades_df: run_backtest_core 回傳的交易明細
        :param market_df: 0050 的 K 線 (fetch_stock_data("0050.TW", ...))
        """
        if trades_df is None or trades_df.empty:
            return pd.DataFrame()
        market_df = add_regime_columns(market_df)
        sig_col = '訊號日' if '訊號日' in trades_df.columns else '日期'
        df = trades_df.copy()
        df['Regime'] = [regime_at(market_df, d) for d in pd.to_datetime(df[sig_col])]
        return df.groupby('Regime')['真實R'].agg(
            交易次數='count', 總淨利R='sum', 平均R='mean',
            勝率=lambda x: (x > 0).mean() * 100).round(2).reset_index()
