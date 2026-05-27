import pandas as pd
import numpy as np
import itertools

class QuantResearchPipeline:
    def __init__(self, df, strategy_class, backtest_engine):
        """
        初始化研究管線
        :param df: 原始 OHLCV 數據
        :param strategy_class: 你的 BaseStrategy 實作 (如 SMCStrategy)
        :param backtest_engine: 負責跑回測並回傳 (CAGR, Sharpe, MaxDD, Trades, Expectancy) 的函數
        """
        self.df = df
        self.strategy = strategy_class()
        self.backtest_engine = backtest_engine
        
    def run_ablation_study(self, base_params):
        """
        1. 特徵消融實驗 (Ablation Study)
        透過真值表 (Truth Table) 開關不同過濾條件，找出 Alpha 的真正來源
        """
        print("啟動特徵消融實驗 (Feature Ablation Study)...")
        
        # 定義我們要測試開關的因子
        features = ['use_sma_filter', 'use_bos', 'use_volume_zscore', 'use_atr_disp']
        
        # 產生所有開關的排列組合 (True/False 真值表)
        combinations = list(itertools.product([True, False], repeat=len(features)))
        
        results = []
        for combo in combinations:
            # 將 True/False 映射到對應的參數名
            param_grid = dict(zip(features, combo))
            
            # 將基礎參數(如 min_gap_pct) 與當前實驗參數合併
            current_params = {**base_params, **param_grid}
            
            # 1. 產生訊號
            signal_df = self.strategy.generate_signals(self.df.copy(), **current_params)
            
            # 2. 送入回測引擎
            metrics = self.backtest_engine(signal_df)
            
            # 3. 記錄結果
            row = {
                'SMA_Filter': param_grid['use_sma_filter'],
                'BOS_Filter': param_grid['use_bos'],
                'Vol_Filter': param_grid['use_volume_zscore'],
                'ATR_Disp': param_grid['use_atr_disp'],
                'Trades': metrics.get('trade_count', 0),
                'WinRate(%)': metrics.get('win_rate', 0),
                'Expectancy(R)': metrics.get('expectancy', 0),
                'Sharpe': metrics.get('sharpe_ratio', 0),
                'MaxDD(%)': metrics.get('max_drawdown', 0),
                'CAGR(%)': metrics.get('cagr', 0)
            }
            results.append(row)
            
        result_df = pd.DataFrame(results)
        # 按夏普值或期望值排序，直觀看出誰在裸泳
        return result_df.sort_values(by='Sharpe', ascending=False)

    def run_regime_analysis(self, signal_df):
        """
        2. 市場環境切割 (Regime Split)
        分析策略在牛/熊、高波動/低波動環境下的表現差異
        """
        print("啟動市場環境切割分析 (Regime Split Analysis)...")
        df = signal_df.copy()
        
        # 定義環境 (Regime)
        # 1. 牛熊切分 (價格在 200MA 之上或之下)
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['Regime_Trend'] = np.where(df['Close'] > df['SMA200'], 'Bull', 'Bear')
        
        # 2. 波動度切分 (ATR 高於或低於過去 200 天中位數)
        df['ATR_14'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min() # 簡化版示意
        df['ATR_Median_200'] = df['ATR_14'].rolling(200).median()
        df['Regime_Vol'] = np.where(df['ATR_14'] > df['ATR_Median_200'], 'High_Vol', 'Low_Vol')
        
        # 組合 Regime
        df['Market_Regime'] = df['Regime_Trend'] + "_" + df['Regime_Vol']
        
        # 只取出有產生交易(FVG==True)的列來分析
        trades_df = df[df['FVG'] == True].copy()
        
        if trades_df.empty:
            return "無交易紀錄可供 Regime 分析"
            
        # 統計各環境下的交易次數與(假設的)單筆 R 報酬
        # 這裡假設你的回測引擎有把每次交易的 'Trade_R_Result' 寫回 df
        if 'Trade_R_Result' in trades_df.columns:
            regime_stats = trades_df.groupby('Market_Regime').agg(
                Trade_Count=('FVG', 'count'),
                Total_R=('Trade_R_Result', 'sum'),
                Avg_R=('Trade_R_Result', 'mean'),
                Win_Rate=('Trade_R_Result', lambda x: (x > 0).mean() * 100)
            ).reset_index()
            return regime_stats
        else:
            # 若無獨立報酬欄位，僅顯示訊號分佈
            return trades_df['Market_Regime'].value_counts().rename("Trade_Count_by_Regime")

# 💡 使用範例 (虛擬碼)：
# pipeline = QuantResearchPipeline(df, SMCStrategy, run_backtest_logic)
# ablation_report = pipeline.run_ablation_study(base_params={'min_gap_pct': 0.3})
# print(ablation_report.to_markdown())