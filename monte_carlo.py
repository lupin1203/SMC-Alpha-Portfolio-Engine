# monte_carlo.py
import numpy as np
import pandas as pd
import logging

def run_monte_carlo(trades_df, num_simulations=1000, ruin_drawdown_pct=30.0, starting_capital=100.0):
    """
    執行蒙地卡羅模擬 (Bootstrap 重抽樣)
    trades_df: 包含 '真實R' 欄位的交易紀錄 DataFrame
    """
    if trades_df.empty or '真實R' not in trades_df.columns:
        logging.warning("沒有足夠的交易紀錄來執行蒙地卡羅模擬。")
        return None

    # 提取所有交易的 R 值
    r_values = trades_df['真實R'].values
    num_trades = len(r_values)
    
    simulated_paths = []
    max_drawdowns = []
    ruin_count = 0

    # 執行 1000 次平行宇宙模擬
    for i in range(num_simulations):
        # 隨機重抽樣 (Bootstrap with replacement)
        random_sequence = np.random.choice(r_values, size=num_trades, replace=True)
        
        # 計算累積 R 值 (資金曲線)
        equity_curve = np.cumsum(random_sequence)
        simulated_paths.append(equity_curve)
        
        # 計算此路徑的最大回撤 (以 R 為單位)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = peak - equity_curve
        max_dd = np.max(drawdown)
        max_drawdowns.append(max_dd)
        
        # 破產判定 (假設每 1R = 預設資金的某個百分比，這裡簡化為直接看 DD_R 是否超過容忍極限)
        # 假設你的 ruin_drawdown_pct 是 30%，單筆風險是 2%，那 15R 的回撤就是破產
        risk_per_trade_pct = 2.0 
        if (max_dd * risk_per_trade_pct) >= ruin_drawdown_pct:
            ruin_count += 1

    # 統計結果
    median_path = np.median(simulated_paths, axis=0)
    worst_path = np.min(simulated_paths, axis=0)
    median_max_dd = np.median(max_drawdowns)
    worst_max_dd = np.max(max_drawdowns)
    prob_of_ruin = (ruin_count / num_simulations) * 100

    results = {
        "num_simulations": num_simulations,
        "median_final_r": median_path[-1],
        "worst_final_r": worst_path[-1],
        "median_max_dd_r": median_max_dd,
        "worst_max_dd_r": worst_max_dd,
        "prob_of_ruin_pct": prob_of_ruin
    }
    
    return results