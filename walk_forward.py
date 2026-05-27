# walk_forward.py
import pandas as pd
import numpy as np
import logging
from strategy import apply_smc_logic
from backtest_engine import run_backtest_core

def run_wfo(df_raw, train_years=3, test_years=1, bt_mode="固定盈虧比", slippage_pct=0.1, fee_disc=0.28, min_vol=1000):
    """
    執行滾動式動態前進分析 (Rolling Walk-Forward Optimization)
    """
    if df_raw.empty or len(df_raw) < 250 * (train_years + test_years):
        logging.warning("資料量不足以執行設定的 WFO 週期。")
        return pd.DataFrame(), None

    # 定義要給機器自動優化的參數網格 (Parameter Grid)
    # 為了運算速度，我們給定幾個核心的 SMC 參數範圍
    gap_options = [0.2, 0.3, 0.4]
    tp_options = [2.0, 3.0, 4.0] if bt_mode == "固定盈虧比" else [8.0, 10.0, 12.0]

    all_oos_trades = []
    
    # 計算時間視窗 (假設一年 252 個交易日)
    train_bars = train_years * 252
    test_bars = test_years * 252
    
    start_idx = 0
    wfo_logs = []

    # 開始履帶式滾動 (Rolling Window)
    while start_idx + train_bars + test_bars <= len(df_raw):
        train_df = df_raw.iloc[start_idx : start_idx + train_bars]
        test_df = df_raw.iloc[start_idx + train_bars : start_idx + train_bars + test_bars]
        
        best_exp = -999
        best_params = {"gap": 0.3, "tp": 3.0}

        # 1. 在 Train 區間尋找最佳參數 (In-Sample Optimization)
        for g in gap_options:
            for tp in tp_options:
                temp_train = apply_smc_logic(train_df.copy(), min_gap_pct=g, use_sma_filter=True)
                res = run_backtest_core(temp_train, bt_mode, tp, slippage_pct, fee_disc, 0.1, min_vol)
                # res[3] 是 Expectancy (期望值)
                if res[1] > 5 and res[3] > best_exp: # 至少要交易 5 次才有代表性
                    best_exp = res[3]
                    best_params = {"gap": g, "tp": tp}
        
        # 2. 將最佳參數套用到完全沒見過的 Test 區間 (Out-of-Sample)
        oos_df_logic = apply_smc_logic(test_df.copy(), min_gap_pct=best_params["gap"], use_sma_filter=True)
        oos_res = run_backtest_core(oos_df_logic, bt_mode, best_params["tp"], slippage_pct, fee_disc, 0.1, min_vol)
        
        # 把這段盲測的交易紀錄存下來
        if len(oos_res[0]) > 0:
            all_oos_trades.append(oos_res[0])
            
        wfo_logs.append({
            "盲測年份": f"{test_df.index[0].date().year}",
            "Train最佳缺口": best_params["gap"],
            "Train最佳停利": best_params["tp"],
            "OOS交易次數": oos_res[1],
            "OOS期望值": round(oos_res[3], 2),
            "OOS淨利(R)": round(oos_res[6], 2)
        })

        # 履帶往前推進一格 (步進 = test_bars)
        start_idx += test_bars

    # 3. 拼接所有盲測期的交易紀錄，形成最終的 WFO 資金曲線
    if all_oos_trades:
        final_oos_df = pd.concat(all_oos_trades).sort_values("日期")
        
        # 重新計算拼接後的總統計數據
        win_rate = len(final_oos_df[final_oos_df['真實R'] > 0]) / len(final_oos_df)
        avg_w = final_oos_df[final_oos_df['真實R'] > 0]['真實R'].mean() if win_rate > 0 else 0
        avg_l = abs(final_oos_df[final_oos_df['真實R'] < 0]['真實R'].mean()) if win_rate < 1 else 1
        final_exp = (win_rate * avg_w) - ((1 - win_rate) * avg_l)
        
        eq_curve = final_oos_df['真實R'].cumsum()
        max_dd = (eq_curve - eq_curve.expanding().max()).min()
        
        final_stats = {
            "總交易次數": len(final_oos_df),
            "總淨利(R)": round(final_oos_df['真實R'].sum(), 2),
            "WFO綜合期望值": round(final_exp, 2),
            "WFO綜合最大回撤": round(max_dd, 2)
        }
        return final_oos_df, pd.DataFrame(wfo_logs), final_stats
    else:
        return pd.DataFrame(), pd.DataFrame(wfo_logs), None