"""用 DannyQuant FAQ 公開的出場 / 部位規則，套在本專案（無前視、還原權息）的 VCP 訊號上。

公開規則（dannyquant.com/faq，2026-11 擷取）：
- 次交易日開盤進場
- 持有 5 個交易日後收盤出場
- 停損 10%（空頭區 8%）；此處「空頭區」= 加權指數 < 200 日均線（他們未定義，屬本專案假設）
- 最多持有 4 檔、每日最多新增 2 檔；資金分 4 份
- 交易成本：來回 0.5425%（他們的假設）

未公開、只能用本專案版本代替的部分：VCP 的具體辨識規則、同日多檔時的排序。
因此這不是「重現」他們的策略，而是檢查：在乾淨的資料上，用同樣的出場與部位規則，
VCP 突破能不能做出接近他們宣稱的績效（VCP 2.1：年化 73.4%、Sharpe 1.80、MDD −39.2%）。

用法：python -m research.vcp_sector.replicate_dq [--no-api]
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import equity_stats, market_regime, simulate_portfolio, simulate_trades, summarize
from .config import DEFAULT_CONFIG
from .lead_lag import StockPanel
from .vcp_detector import detect_all

DQ_TRADE = replace(DEFAULT_CONFIG.trade, exit_mode="fixed_close", fixed_hold_days=5,
                   max_positions=4, max_new_per_day=2, sizing="equal", lot_size=1)
COSTS = {
    "DQ 假設成本 0.5425%": dict(fee_rate=0.0012125, tax_rate=0.003, slippage=0.0),
    "實際成本 ≈0.885%（含滑價）": dict(fee_rate=0.001425, tax_rate=0.003, slippage=0.001),
}


def dq_signals(vcp: pd.DataFrame, bench: pd.Series, label: str) -> pd.DataFrame:
    bo = vcp[vcp["is_breakout"]]
    reg = market_regime(bench).reindex(pd.DatetimeIndex(bo["date"]), method="ffill").to_numpy()
    stop_pct = np.where(pd.Series(reg).str.startswith("Bear"), 0.08, 0.10)
    return pd.DataFrame({"signal_date": bo["date"].to_numpy(), "stock_id": bo["stock_id"].to_numpy(),
                         "group": "", "variant": label, "stop_price": np.nan, "stop_pct": stop_pct})


def random_signals(panel: StockPanel, sig: pd.DataFrame, rng) -> pd.DataFrame:
    elig = panel.eligible.to_numpy()
    rows = []
    for d, g in sig.groupby("signal_date"):
        pool = np.flatnonzero(elig[panel.row[d]])
        for j, sp in zip(rng.choice(pool, size=min(len(g), len(pool)), replace=False), g["stop_pct"]):
            rows.append({"signal_date": d, "stock_id": panel.stocks[j], "group": "", "variant": "random",
                         "stop_price": np.nan, "stop_pct": sp})
    return pd.DataFrame(rows)


def run(panel, sig, tcfg):
    tr = simulate_trades(panel, sig, tcfg)
    eq, taken = simulate_portfolio(panel, tr, tcfg)
    return tr, eq, taken


def main(argv=None):
    from .report import OUTPUT_DIR, load_inputs

    p = argparse.ArgumentParser()
    p.add_argument("--no-api", action="store_true")
    p.add_argument("--n-random", type=int, default=20)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 220)
    cfg = DEFAULT_CONFIG

    inp = load_inputs("official", False, not args.no_api, cfg)
    panel = StockPanel(inp["prices"], inp["eligible"], cfg.leadlag, cfg.data.turnover_window)
    bench = inp["benchmark"]

    variants = {
        "VCP 嚴格（本專案現行規則）": cfg.vcp,
        "VCP 寬鬆（無整理期/pivot 限制）": replace(cfg.vcp, min_base_days=0, max_pivot_below_base_high=1.0),
    }
    sigs = {k: dq_signals(detect_all(panel, v, setup_on_last_day=False), bench, k) for k, v in variants.items()}

    rows, yearly = [], {}
    for cost_name, cost in COSTS.items():
        tcfg = replace(DQ_TRADE, **cost)
        for name, sig in sigs.items():
            tr, eq, taken = run(panel, sig, tcfg)
            st = equity_stats(eq)
            sm = summarize(tr)
            rows.append({"成本": cost_name, "訊號": name, "訊號數": len(sig), "成交筆數": len(taken),
                         "勝率": sm.get("win_rate"), "平均單筆報酬": sm.get("avg_ret"),
                         "年化": st.get("cagr"), "最大回撤": st.get("max_drawdown"), "Sharpe": st.get("sharpe"),
                         "10年倍數": eq.iloc[-1] / eq.iloc[0] if len(eq) else np.nan})
            if cost_name.startswith("實際"):
                yearly[name] = eq.resample("YE").last().pct_change().fillna(eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1)
            # 隨機進場：同日同筆數、同規則
            rng = np.random.default_rng(0)
            cagrs, rets = [], []
            for _ in range(args.n_random):
                rtr, req, _ = run(panel, random_signals(panel, sig, rng), tcfg)
                cagrs.append(equity_stats(req).get("cagr"))
                rets.append(rtr["ret"].mean())
            rows.append({"成本": cost_name, "訊號": "　└ 同日隨機選股 ×%d（平均）" % args.n_random,
                         "平均單筆報酬": np.mean(rets), "年化": np.mean(cagrs),
                         "最大回撤": np.nan, "Sharpe": np.nan})

    res = pd.DataFrame(rows)
    b = bench.loc[sigs[next(iter(sigs))]["signal_date"].min():]
    print("\n=== 用 DannyQuant 公開規則（5 日收盤出場、10%/8% 停損、4 檔、每日最多新增 2 檔）===")
    print("DQ 宣稱 VCP 2.1：年化 73.4%、Sharpe 1.80、最大回撤 −39.2%（2016–2026）\n")
    fmt = {"勝率": "{:.1%}", "平均單筆報酬": "{:+.2%}", "年化": "{:+.1%}", "最大回撤": "{:.1%}",
           "Sharpe": "{:.2f}", "10年倍數": "{:.2f}x"}
    out = res.copy()
    for c, f in fmt.items():
        out[c] = out[c].map(lambda x, f=f: "" if pd.isna(x) else f.format(x))
    for c in ("訊號數", "成交筆數"):
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{int(x)}")
    print(out.to_string(index=False))
    print(f"\n加權指數（價格，不含息）同期：年化 {equity_stats(b).get('cagr'):+.1%}、"
          f"最大回撤 {equity_stats(b).get('max_drawdown'):.1%}")
    print("\n逐年報酬（實際成本）：")
    print(pd.DataFrame(yearly).set_axis(pd.DataFrame(yearly).index.year).to_string(
        float_format=lambda x: f"{x:+.1%}"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTPUT_DIR / "replicate_dq.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
