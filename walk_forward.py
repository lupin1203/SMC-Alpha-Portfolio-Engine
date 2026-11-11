# walk_forward.py
"""
滾動式前進分析 (Rolling Walk-Forward Optimization)，單一股票用。

每一段：用前 train_years 年挑參數 (期望值最高、且至少 MIN_TRAIN_TRADES 筆交易)，
再把挑出的參數套用在接下來 test_years 年 (完全沒看過的資料)，最後把所有盲測段接起來。
"""
import logging

import pandas as pd

from strategy import apply_smc_logic
from backtest_engine import run_backtest_core, summarize_trades

MIN_TRAIN_TRADES = 5   # 訓練期至少要有幾筆交易，挑出的參數才算數
WARMUP_BARS = 60       # 盲測段前面多帶幾根 K 棒，讓季線等指標在盲測第一天就有值


def run_wfo(df_raw, train_years=3, test_years=1, bt_mode="固定盈虧比", slippage_pct=0.002,
            fee_disc=0.28, tax_rate=0.003, min_vol=0, use_sma_filter=True, realistic=True):
    """
    回傳固定 3 個值：(盲測交易明細, 每段紀錄, 總統計 dict 或 None)。
    slippage_pct 是比例 (0.002 = 0.2%)。realistic=True 用實盤真實模式成交。
    """
    train_bars = train_years * 252
    test_bars = test_years * 252
    if df_raw is None or df_raw.empty or len(df_raw) < train_bars + test_bars:
        logging.warning("資料量不足以執行設定的 WFO 週期。")
        return pd.DataFrame(), pd.DataFrame(), None

    gap_options = [0.2, 0.3, 0.4]
    tp_options = [2.0, 3.0, 4.0] if bt_mode == "固定盈虧比" else [8.0, 10.0, 12.0]

    def backtest(df, gap, tp):
        logic = apply_smc_logic(df.copy(), min_gap_pct=gap, use_sma_filter=use_sma_filter)
        trades, _ = run_backtest_core(logic, bt_mode, tp, slippage_pct, fee_disc, tax_rate, min_vol,
                                      realistic=realistic)
        return trades

    all_oos_trades = []
    wfo_logs = []
    start_idx = 0

    while start_idx + train_bars + test_bars <= len(df_raw):
        train_end = start_idx + train_bars
        train_df = df_raw.iloc[start_idx:train_end]
        test_start_ts = df_raw.index[train_end]
        # 盲測段往前多帶一些 K 棒暖機，之後只保留訊號日落在盲測段內的交易
        test_df = df_raw.iloc[max(train_end - WARMUP_BARS, 0):train_end + test_bars]

        # 1. 訓練期挑參數
        best_exp = None
        best_params = {"gap": 0.3, "tp": tp_options[1]}
        for g in gap_options:
            for tp in tp_options:
                s = summarize_trades(backtest(train_df, g, tp))
                if s['trades'] >= MIN_TRAIN_TRADES and (best_exp is None or s['expectancy'] > best_exp):
                    best_exp = s['expectancy']
                    best_params = {"gap": g, "tp": tp}

        # 2. 盲測期套用
        oos_trades = backtest(test_df, best_params["gap"], best_params["tp"])
        if not oos_trades.empty:
            sig_col = '訊號日' if '訊號日' in oos_trades.columns else '日期'
            oos_trades = oos_trades[pd.to_datetime(oos_trades[sig_col]) >= test_start_ts]
        oos = summarize_trades(oos_trades)
        if oos['trades'] > 0:
            all_oos_trades.append(oos_trades)

        wfo_logs.append({
            "盲測年份": str(pd.Timestamp(test_start_ts).year),
            "Train最佳缺口": best_params["gap"],
            "Train最佳停利": best_params["tp"],
            "Train期望值": round(best_exp, 2) if best_exp is not None else None,
            "OOS交易次數": oos['trades'],
            "OOS期望值": round(oos['expectancy'], 2),
            "OOS淨利(R)": round(oos['total_r'], 2),
        })

        start_idx += test_bars

    logs_df = pd.DataFrame(wfo_logs)
    if not all_oos_trades:
        return pd.DataFrame(), logs_df, None

    final_oos_df = pd.concat(all_oos_trades).sort_values("日期").reset_index(drop=True)
    total = summarize_trades(final_oos_df)
    final_stats = {
        "總交易次數": total['trades'],
        "總淨利(R)": round(total['total_r'], 2),
        "WFO綜合期望值": round(total['expectancy'], 2),
        "WFO綜合最大回撤(R)": round(total['max_dd_r'], 2),
    }
    return final_oos_df, logs_df, final_stats
