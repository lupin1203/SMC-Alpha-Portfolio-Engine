# strategy.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df, **kwargs):
        pass

class SMCStrategy(BaseStrategy):
    def generate_signals(self, df, min_gap_pct=0.0, use_sma_filter=False, **kwargs):
        # 🌟 透過 kwargs 動態接收，不怕參數報錯
        use_vol_filter = kwargs.get('use_vol_filter', False)
        use_atr_filter = kwargs.get('use_atr_filter', False)
        use_bos_filter = kwargs.get('use_bos_filter', False)
        # 進場方式："fvg" (預設，原策略) / "breakout20" (收盤突破前 20 日最高價，對照用)
        entry_type = kwargs.get('entry_type', 'fvg')
        
        if len(df) < 60:
            df['FVG'] = False
            df['adjusted_risk'] = 0.0
            return df
            
        df['SMA60'] = df['Close'].rolling(60).mean()

        if entry_type == 'breakout20':
            # 對照組：單純突破策略。停損距離用與 FVG 相同的下限 (收盤價 3%)，出場規則不變
            prior_high_20 = df['High'].shift(1).rolling(20).max()
            signal = df['Close'] > prior_high_20
            if use_sma_filter:
                signal = signal & (df['Close'] > df['SMA60'])
            df['FVG'] = signal.fillna(False)
            df['adjusted_risk'] = df['Close'] * 0.03
            return df
        elif entry_type != 'fvg':
            raise ValueError(f"未知的 entry_type: {entry_type}")
        
        fvg_cond = (df['Low'] > df['High'].shift(2))
        df['gap_size_pct'] = (df['Low'] - df['High'].shift(2)) / df['High'].shift(2) * 100
        gap_filter = df['gap_size_pct'] >= min_gap_pct
        
        yesterday_body = df['Close'].shift(1) - df['Open'].shift(1)
        bullish_base = (yesterday_body > 0)
        
        if use_atr_filter:
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift(1)).abs()
            low_close = (df['Low'] - df['Close'].shift(1)).abs()
            df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            bullish_expansion = bullish_base & (yesterday_body >= (df['ATR'].shift(1) * 0.3))
        else:
            bullish_expansion = bullish_base 

        if use_vol_filter and 'Volume' in df.columns:
            df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
            volume_conf = df['Volume'].shift(1) > df['Vol_SMA20'].shift(1)
        else:
            volume_conf = True 
            
        if use_bos_filter:
            df['Swing_High_20'] = df['High'].shift(2).rolling(window=20, min_periods=5).max()
            bos_conf = (df['Close'].shift(1) > df['Swing_High_20']) | (df['Close'] > df['Swing_High_20'])
        else:
            bos_conf = True 
        
        df['risk_dist'] = df['Low'] - df['High'].shift(2)
        df['min_risk_allowed'] = df['Close'] * 0.03
        
        # 計算初步風險
        temp_risk = df[['risk_dist', 'min_risk_allowed']].max(axis=1)
        
        # 🌟 強制防呆：如果風險為 0 或小於 0，強制設為股價的 5%
        df['adjusted_risk'] = np.where(temp_risk <= 0, df['Close'] * 0.05, temp_risk)
        gap_up_limit = (df['Open'] < df['Close'].shift(1) * 1.05)
        fvg_core = fvg_cond & gap_filter & bullish_expansion & volume_conf & bos_conf & gap_up_limit
        
        if use_sma_filter:
            df['FVG'] = fvg_core & (df['Close'] > df['SMA60'])
        else:
            df['FVG'] = fvg_core
            
        df['FVG'] = df['FVG'].fillna(False)
        return df

def apply_smc_logic(df, min_gap_pct, use_sma_filter, **kwargs):
    strategy = SMCStrategy()
    return strategy.generate_signals(
        df, 
        min_gap_pct=min_gap_pct, 
        use_sma_filter=use_sma_filter,
        **kwargs
    )

def randomize_signals(df, seed=0, salt="", warmup=60, candidate_mask=None):
    """
    隨機進場對照組：保留原本的訊號「數量」，但把訊號日期隨機打散到其他交易日，
    停損距離改用與原策略相同的下限規則 (收盤價 3%)。
    出場規則完全不變，所以若隨機版績效與原策略差不多，代表 FVG 進場訊號本身沒有貢獻。

    seed + salt (例如股票代號) 決定亂數，確保結果可重現、且跨進程一致。
    candidate_mask: 只在這些交易日裡隨機挑 (例如只挑收盤在季線上的日子)；None = 所有日子。
    """
    import zlib
    df = df.copy()
    n_signals = int(df['FVG'].fillna(False).sum()) if 'FVG' in df.columns else 0
    df['FVG'] = False
    if n_signals == 0 or len(df) <= warmup:
        return df
    rng = np.random.default_rng(zlib.crc32(f"{salt}|{seed}".encode("utf-8")))
    candidates = np.arange(warmup, len(df))
    if candidate_mask is not None:
        mask = np.asarray(pd.Series(candidate_mask, index=df.index).fillna(False), dtype=bool)
        candidates = candidates[mask[warmup:]]
        if len(candidates) == 0:
            return df
    picks = rng.choice(candidates, size=min(n_signals, len(candidates)), replace=False)
    fvg = np.zeros(len(df), dtype=bool)
    fvg[picks] = True
    df['FVG'] = fvg
    df['adjusted_risk'] = df['Close'] * 0.03
    return df
