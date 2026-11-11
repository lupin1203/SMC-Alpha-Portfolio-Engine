# monte_carlo.py
import numpy as np
import pandas as pd
import logging

def run_monte_carlo(trades_df, num_simulations=1000, ruin_drawdown_pct=30.0, starting_capital=100.0, risk_per_trade_pct=0.5):
    """
    執行蒙地卡羅模擬 (Bootstrap 重抽樣)
    trades_df: 包含 '真實R' 欄位的交易紀錄 DataFrame
    risk_per_trade_pct: 單筆風險百分比 (e.g. 2.0 代表 2%)，用於計算破產門檻
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
        
        # 破產判定：用傳入的單筆風險計算實際 DD%，超過容忍極限即視為破產
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

# ==========================================
# 歷史黑天鵝場景壓力測試
# ==========================================

# 已知的台股重大事件：(名稱, 描述, 連續虧損序列的 R 倍數估算)
HISTORICAL_SCENARIOS = {
    "2020_covid": {
        "name": "2020 COVID 崩盤",
        "desc": "2020/3，台股10天跌近30%",
        "shock_r": -8.0,       # 單月最大衝擊（R）
        "duration_days": 20,   # 持續天數
        "recovery_days": 60,   # 恢復天數
    },
    "2022_rate_hike": {
        "name": "2022 升息熊市",
        "desc": "2022全年台股跌幅約30%，長達9個月",
        "shock_r": -3.0,
        "duration_days": 180,
        "recovery_days": 120,
    },
    "2015_china_crash": {
        "name": "2015 中國股災",
        "desc": "2015/8，台股單週跌幅10%以上",
        "shock_r": -5.0,
        "duration_days": 10,
        "recovery_days": 30,
    },
    "2008_gfc": {
        "name": "2008 金融海嘯",
        "desc": "台股從高點跌幅近60%，持續約1年",
        "shock_r": -4.0,
        "duration_days": 240,
        "recovery_days": 360,
    },
}


def run_scenario_stress_test(
    trades_df: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    risk_per_trade_pct: float = 0.5,
) -> dict:
    """
    歷史黑天鵝場景壓力測試。
    對每個場景，在隨機位置插入衝擊序列，計算帳戶淨值影響。

    回傳：{scenario_name: {"min_equity": float, "max_dd_pct": float, "survived": bool}}
    """
    if trades_df.empty or '真實R' not in trades_df.columns:
        return {}

    r_values = trades_df['真實R'].values
    risk_frac = risk_per_trade_pct / 100.0
    results   = {}

    for key, scenario in HISTORICAL_SCENARIOS.items():
        shock_r        = scenario["shock_r"]
        duration_days  = scenario["duration_days"]
        # 估算衝擊期間觸發多少筆交易（假設平均持倉 10 天）
        shock_trades   = max(1, duration_days // 10)
        shock_sequence = [shock_r] * shock_trades

        # 在 r_values 中隨機插入衝擊序列，跑 200 次取最差情況
        worst_dd  = 0.0
        min_eq    = initial_capital
        n_runs    = 200

        for _ in range(n_runs):
            insert_at = np.random.randint(0, max(1, len(r_values)))
            combined  = np.concatenate([r_values[:insert_at], shock_sequence, r_values[insert_at:]])

            equity = initial_capital
            peak   = initial_capital
            run_min_eq = initial_capital

            for r in combined:
                equity += equity * risk_frac * r
                equity  = max(equity, 0)
                peak    = max(peak, equity)
                run_min_eq = min(run_min_eq, equity)
                if equity <= 0:
                    break

            dd_pct = (peak - run_min_eq) / peak * 100 if peak > 0 else 100
            worst_dd  = max(worst_dd, dd_pct)
            min_eq    = min(min_eq, run_min_eq)

        results[key] = {
            "name":        scenario["name"],
            "desc":        scenario["desc"],
            "min_equity":  round(min_eq, 0),
            "max_dd_pct":  round(worst_dd, 1),
            "survived":    min_eq > 0,
        }

    return results