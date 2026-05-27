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
        
        if len(df) < 60:
            df['FVG'] = False
            df['adjusted_risk'] = 0.0
            return df
            
        df['SMA60'] = df['Close'].rolling(60).mean()
        
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