"""Phase 8：回測 + walk-forward。

用法（repo 根目錄）::

    python -m research.vcp_sector.backtest                 # 訊號 B，官方產業別
    python -m research.vcp_sector.backtest --groups theme
    python -m research.vcp_sector.backtest --synthetic     # 人造資料，只檢查流程

時間軸（防前視）：
- 訊號在 t 日收盤後產生（只用 t 日含以前資料），t+1 日開盤進場
- 停損價在 t 日就決定；盤中觸及停損 → 以 min(開盤, 停損) 出場（跳空跌破用開盤價）
- 移動停利 / 跌破 MA20 / 時間停損都是 d 日收盤判斷、d+1 日開盤出場
- walk-forward 的 IS 只計入「出場日 < OOS 起始日」的交易，避免 IS 用到 OOS 期間的價格
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, Config, TradeConfig
from .lead_lag import StockPanel
from .sector_strength import apply_strong_rule
from .signals import lowest_low_stop, signal_b

logger = logging.getLogger(__name__)

TRADE_COLS = ["signal_date", "stock_id", "group", "variant", "entry_date", "entry_price", "stop_price",
              "exit_date", "exit_price", "exit_reason", "hold_days", "ret", "r"]


# ---------------------------------------------------------------------------
# 單筆交易模擬
# ---------------------------------------------------------------------------
def simulate_trades(panel: StockPanel, signals: pd.DataFrame,
                    tcfg: TradeConfig = DEFAULT_CONFIG.trade) -> pd.DataFrame:
    """逐筆模擬。同一 variant 同一檔若前一筆尚未出場，新訊號略過。"""
    O, H, L, C = (getattr(panel, f).to_numpy() for f in ("open", "high", "low", "close"))
    ma = panel.close.rolling(tcfg.exit_ma, min_periods=tcfg.exit_ma).mean().to_numpy()
    dates = panel.dates
    n = len(dates)
    cost_in = 1 + tcfg.fee_rate
    cost_out = 1 - tcfg.fee_rate - tcfg.tax_rate

    busy_until: dict[tuple, int] = {}
    out = []
    for sig in signals.sort_values(["signal_date", "stock_id"]).itertuples(index=False):
        i = panel.row.get(sig.signal_date)
        j = panel.col.get(sig.stock_id)
        if i is None or j is None or i + 1 >= n:
            continue
        key = (sig.variant, sig.stock_id)
        if busy_until.get(key, -1) >= i:
            continue
        e = i + 1
        stop = sig.stop_price
        if not np.isfinite(O[e, j]) or not np.isfinite(stop):
            continue
        entry = O[e, j] * (1 + tcfg.slippage)
        risk = (entry - stop) / entry
        if not (tcfg.min_risk_pct <= risk <= tcfg.max_risk_pct):
            continue

        hi = -np.inf
        pending = None
        exit_px, reason, d_exit = np.nan, None, None
        for d in range(e, n):
            if not np.isfinite(C[d, j]):           # 停牌
                continue
            if pending is not None and d > e:
                exit_px, reason, d_exit = O[d, j], pending, d
                break
            if L[d, j] <= stop:
                exit_px = stop if d == e else min(O[d, j], stop)
                reason, d_exit = "stop", d
                break
            hi = max(hi, C[d, j])
            if C[d, j] < hi * (1 - tcfg.trail_pct):
                pending = "trail"
            elif d > 0 and np.isfinite(ma[d, j]) and C[d, j] < ma[d, j] and C[d - 1, j] >= ma[d - 1, j]:
                pending = "ma_break"
            elif d - e + 1 >= tcfg.max_hold_days:
                pending = "time"
        if d_exit is None:                          # 資料結束仍持有：以最後收盤計
            valid = np.where(np.isfinite(C[e:, j]))[0]
            d_exit = e + valid[-1]
            exit_px, reason = C[d_exit, j], "open"

        exit_net = exit_px * (1 - tcfg.slippage) * cost_out
        entry_cost = entry * cost_in
        out.append({
            "signal_date": sig.signal_date, "stock_id": sig.stock_id, "group": sig.group,
            "variant": sig.variant, "entry_date": dates[e], "entry_price": entry, "stop_price": stop,
            "exit_date": dates[d_exit], "exit_price": exit_px, "exit_reason": reason,
            "hold_days": d_exit - e + 1, "ret": exit_net / entry_cost - 1,
            "r": (exit_net - entry_cost) / (entry - stop),
        })
        busy_until[key] = d_exit
    return pd.DataFrame(out, columns=TRADE_COLS)


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def summarize(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series({"n": 0})
    r = trades["r"]
    wins, losses = r[r > 0], r[r <= 0]
    return pd.Series({
        "n": len(r),
        "win_rate": (r > 0).mean(),
        "avg_r": r.mean(),
        "median_r": r.median(),
        "avg_win_r": wins.mean() if len(wins) else np.nan,
        "avg_loss_r": losses.mean() if len(losses) else np.nan,
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.nan,
        "avg_ret": trades["ret"].mean(),
        "avg_hold": trades["hold_days"].mean(),
        "t_stat": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan,
    })


def by_group(trades: pd.DataFrame, key) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby(key).apply(summarize, include_groups=False)


def market_regime(benchmark: pd.Series, vol_threshold: float = 25.0) -> pd.Series:
    """與 SMC 系統相同的四格 regime：Bull/Bear（> SMA200）× High/Low Vol（20 日年化波動 > 25%）。"""
    b = benchmark.sort_index()
    sma200 = b.rolling(200, min_periods=200).mean()
    vol = b.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252) * 100
    reg = np.where(b > sma200, "Bull", "Bear")
    reg = pd.Series(reg, index=b.index) + "_" + np.where(vol > vol_threshold, "HighVol", "LowVol")
    return reg.where(sma200.notna() & vol.notna(), "Unknown")


def attach_regime(trades: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    reg = market_regime(benchmark)
    t = trades.copy()
    t["regime"] = reg.reindex(pd.DatetimeIndex(t["signal_date"]), method="ffill").to_numpy()
    return t


# ---------------------------------------------------------------------------
# 投組模擬（固定風險 1%）
# ---------------------------------------------------------------------------
def simulate_portfolio(panel: StockPanel, trades: pd.DataFrame,
                       tcfg: TradeConfig = DEFAULT_CONFIG.trade) -> tuple[pd.Series, pd.DataFrame]:
    """逐日盯市的投組。同日多筆訊號依 stock_id 排序先到先得（不偷看結果）。

    回傳 (每日權益曲線, 實際成交的交易)。
    """
    if trades.empty:
        return pd.Series(dtype=float), trades
    C = panel.close.ffill().to_numpy()
    cost_in = 1 + tcfg.fee_rate
    cost_out = 1 - tcfg.fee_rate - tcfg.tax_rate
    entries = {d: g for d, g in trades.sort_values("stock_id").groupby("entry_date")}

    cash = tcfg.initial_capital
    positions: list[dict] = []
    taken = []
    if (trades["exit_date"] < trades["entry_date"]).any():
        raise ValueError("exit_date 早於 entry_date")
    equity = np.full(len(panel.dates), np.nan)
    start = panel.row[trades["entry_date"].min()]
    for k in range(start, len(panel.dates)):
        d = panel.dates[k]
        # 1) 開盤：上一日收盤觸發、今日開盤出場的部位先出（exit_date == d 且非當日進場者）
        positions = _close_positions(positions, d, tcfg, cost_out, cash_box := [cash])
        cash = cash_box[0]
        eq_open = cash + sum(p["shares"] * C[k - 1 if k else k, p["j"]] for p in positions)
        # 2) 開盤進場
        if d in entries:
            for t in entries[d].itertuples(index=False):
                if len(positions) >= tcfg.max_positions:
                    break
                risk_ps = t.entry_price - t.stop_price
                shares = tcfg.risk_per_trade * eq_open / risk_ps
                shares = min(shares, tcfg.max_position_pct * eq_open / t.entry_price,
                             cash / (t.entry_price * cost_in))
                shares = np.floor(shares / tcfg.lot_size) * tcfg.lot_size
                if shares <= 0:
                    continue
                cash -= shares * t.entry_price * cost_in
                positions.append({"j": panel.col[t.stock_id], "shares": shares,
                                  "exit_date": t.exit_date, "exit_price": t.exit_price})
                taken.append(t._asdict() | {"shares": shares})
        # 3) 盤中：當日觸及停損（含進場當日就停損）的部位出場
        positions = _close_positions(positions, d, tcfg, cost_out, cash_box := [cash])
        cash = cash_box[0]
        equity[k] = cash + sum(p["shares"] * C[k, p["j"]] for p in positions)
    eq = pd.Series(equity, index=panel.dates).dropna()
    return eq, pd.DataFrame(taken)


def _close_positions(positions, d, tcfg, cost_out, cash_box):
    still = []
    for p in positions:
        if p["exit_date"] <= d:
            cash_box[0] += p["shares"] * p["exit_price"] * (1 - tcfg.slippage) * cost_out
        else:
            still.append(p)
    if any(p["exit_date"] < d for p in positions):
        raise AssertionError("部位出場日早於當日，投組模擬順序錯誤")
    return still


def equity_stats(eq: pd.Series) -> pd.Series:
    if eq.empty:
        return pd.Series(dtype=float)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    ret = eq.pct_change().dropna()
    return pd.Series({
        "final_equity": eq.iloc[-1],
        "cagr": (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan,
        "max_drawdown": dd.min(),
        "sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan,
    })


# ---------------------------------------------------------------------------
# 隨機進場 baseline
# ---------------------------------------------------------------------------
def random_baseline(panel: StockPanel, signals: pd.DataFrame, tcfg: TradeConfig = DEFAULT_CONFIG.trade,
                    n_runs: int = 20, seed: int = 0) -> pd.DataFrame:
    """同一天、同樣筆數，改從「當日全市場可交易股」隨機抽，套用完全相同的停損 / 出場規則。"""
    rng = np.random.default_rng(seed)
    stop_np = lowest_low_stop(panel, tcfg.stop_lookback).to_numpy()
    elig_np = panel.eligible.to_numpy()
    stocks = np.asarray(panel.stocks)
    counts = signals.groupby(["signal_date", "variant"]).size()
    rows = []
    for run in range(n_runs):
        picks = []
        for (d, variant), k in counts.items():
            i = panel.row.get(d)
            if i is None:
                continue
            pool = np.flatnonzero(elig_np[i])
            if len(pool) == 0:
                continue
            for j in rng.choice(pool, size=min(k, len(pool)), replace=False):
                picks.append({"signal_date": d, "stock_id": stocks[j], "group": "random",
                              "variant": variant, "stop_price": stop_np[i, j]})
        tr = simulate_trades(panel, pd.DataFrame(picks), tcfg)
        for variant, g in tr.groupby("variant"):
            rows.append({"run": run, "variant": variant, "n": len(g), "avg_r": g["r"].mean(),
                         "win_rate": (g["r"] > 0).mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 敏感度 + walk-forward
# ---------------------------------------------------------------------------
def grid_trades(panel, strength, groups, cfg: Config = DEFAULT_CONFIG) -> dict[tuple, pd.DataFrame]:
    """對族群門檻小網格，各自產生訊號 B 並模擬。key = (pct_above_ma20, rank)。"""
    out = {}
    cache: dict = {}   # (date, group) → (leaders, laggards)，與門檻無關，網格共用
    for pa in cfg.wf.grid_pct_above_ma20:
        for rk in cfg.wf.grid_rank:
            scfg = replace(cfg.sector, strong_pct_above_ma20=pa, strong_rank=rk)
            st = apply_strong_rule(strength, scfg)
            sig = signal_b(panel, st, groups, cfg.leadlag, cfg.trade, cache=cache)
            out[(pa, rk)] = simulate_trades(panel, sig, cfg.trade)
    return out


def sensitivity_table(grid: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (pa, rk), tr in grid.items():
        for variant, g in tr.groupby("variant"):
            rows.append({"pct_above_ma20": pa, "rank": rk, "variant": variant, **summarize(g).to_dict()})
    return pd.DataFrame(rows)


def walk_forward(grid: dict[tuple, pd.DataFrame], fixed_key: tuple,
                 cfg: Config = DEFAULT_CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IS 3 年選參數（依 IS 平均 R），OOS 1 年驗證，逐年滾動。

    同時列出「固定預設參數」在同一 OOS 年的表現，兩者都只看 OOS。
    回傳 (每個窗口的明細, 串接後的 OOS 交易)。
    """
    all_tr = pd.concat([t.assign(key=str(k)) for k, t in grid.items()], ignore_index=True)
    fixed_key = str(fixed_key)
    if all_tr.empty:
        return pd.DataFrame(), pd.DataFrame()
    years = sorted(all_tr["signal_date"].dt.year.unique())
    rows, oos_parts = [], []
    for variant in sorted(all_tr["variant"].unique()):
        tv = all_tr[all_tr["variant"] == variant]
        for y in years[cfg.wf.is_years:]:
            is_start, oos_start = pd.Timestamp(y - cfg.wf.is_years, 1, 1), pd.Timestamp(y, 1, 1)
            oos_end = pd.Timestamp(y + cfg.wf.oos_years, 1, 1)
            is_tr = tv[(tv["signal_date"] >= is_start) & (tv["exit_date"] < oos_start)]
            oos_tr = tv[(tv["signal_date"] >= oos_start) & (tv["signal_date"] < oos_end)]
            perf = is_tr.groupby("key")["r"].agg(["mean", "size"])
            perf = perf[perf["size"] >= cfg.wf.min_is_trades]
            if perf.empty:
                continue
            best = perf["mean"].idxmax()
            sel = oos_tr[oos_tr["key"] == best]
            fix = oos_tr[oos_tr["key"] == fixed_key]
            rows.append({"variant": variant, "oos_year": y, "chosen": best,
                         "is_avg_r": perf.loc[best, "mean"], "is_n": int(perf.loc[best, "size"]),
                         "oos_avg_r": sel["r"].mean(), "oos_n": len(sel),
                         "fixed_oos_avg_r": fix["r"].mean(), "fixed_oos_n": len(fix)})
            oos_parts.append(sel.assign(mode="wf_selected"))
            oos_parts.append(fix.assign(mode="fixed"))
    return pd.DataFrame(rows), (pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.3f}")


def main(argv=None):
    from .report import OUTPUT_DIR, load_inputs

    p = argparse.ArgumentParser(description="訊號 B 回測（族群剛轉強 → 龍頭 / 落後）")
    p.add_argument("--groups", choices=["official", "theme"], default="official")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no-api", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 200)
    cfg = DEFAULT_CONFIG

    inp = load_inputs(args.groups, args.synthetic, not args.no_api, cfg)
    panel = StockPanel(inp["prices"], inp["eligible"], cfg.leadlag, cfg.data.turnover_window)
    strength, groups, bench = inp["strength"], inp["groups"], inp["benchmark"]

    logger.info("產生訊號 B 並模擬（含 %d 組敏感度網格）…",
                len(cfg.wf.grid_pct_above_ma20) * len(cfg.wf.grid_rank))
    grid = grid_trades(panel, strength, groups, cfg)
    fixed_key = (cfg.sector.strong_pct_above_ma20, cfg.sector.strong_rank)
    trades = attach_regime(grid[fixed_key], bench)

    tag = ("synthetic_" if args.synthetic else "") + args.groups
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUTPUT_DIR / f"trades_B_{tag}.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"訊號 B｜固定參數 pct_above_ma20>={fixed_key[0]}、rank>={fixed_key[1]}｜"
          f"{trades['signal_date'].min():%Y-%m-%d} ~ {trades['exit_date'].max():%Y-%m-%d}")
    print("=" * 80)
    print("\n[1] 單筆交易統計（R = 以停損距離為 1R，已扣手續費、證交稅、滑價）")
    print(_fmt(by_group(trades, "variant")))

    print("\n[2] 隨機進場 baseline（同日同筆數，全市場可交易股隨機抽，同樣出場規則）")
    sig_fixed = trades[["signal_date", "stock_id", "group", "variant", "stop_price"]]
    base = random_baseline(panel, sig_fixed, cfg.trade, cfg.wf.n_random_baseline)
    for variant, g in base.groupby("variant"):
        actual = trades.loc[trades["variant"] == variant, "r"].mean()
        pct = (g["avg_r"] < actual).mean()
        print(f"  {variant:10s} 實際 avg_r={actual:+.3f}｜隨機 avg_r 平均 {g['avg_r'].mean():+.3f}"
              f"（範圍 {g['avg_r'].min():+.3f} ~ {g['avg_r'].max():+.3f}）｜實際勝過 {pct:.0%} 的隨機組")

    cols = ["n", "win_rate", "avg_r", "t_stat"]
    print("\n[3] 分年度（看是否只在趨勢年賺錢）")
    yearly = (trades.assign(year=trades["signal_date"].dt.year)
              .groupby(["year", "variant"]).apply(summarize, include_groups=False)[cols].unstack("variant"))
    yearly.columns = [f"{v}:{m}" for m, v in yearly.columns]
    print(_fmt(yearly[sorted(yearly.columns)]))

    print("\n[4] 分 regime（訊號日大盤狀態）")
    print(_fmt(trades.groupby(["regime", "variant"]).apply(summarize, include_groups=False)[cols]))

    print("\n[5] 參數敏感度（族群門檻小網格；看穩定性，不是挑最高分）")
    sens = sensitivity_table(grid)
    sens.to_csv(OUTPUT_DIR / f"sensitivity_B_{tag}.csv", index=False, encoding="utf-8-sig")
    print(_fmt(sens.pivot_table(index=["variant", "pct_above_ma20"], columns="rank", values="avg_r")))

    print("\n[6] Walk-forward（IS 3 年選參數 → OOS 1 年；只看 OOS）")
    wf, oos = walk_forward(grid, fixed_key, cfg)
    if not wf.empty:
        wf.to_csv(OUTPUT_DIR / f"walkforward_B_{tag}.csv", index=False, encoding="utf-8-sig")
        print(_fmt(wf))
        agg = oos.groupby(["variant", "mode"]).apply(summarize, include_groups=False)
        print("\n  OOS 串接：")
        print(_fmt(agg[["n", "win_rate", "avg_r", "t_stat"]]))
        pos = wf.groupby("variant")["fixed_oos_avg_r"].apply(lambda s: f"{(s > 0).sum()}/{len(s)}")
        print("\n  固定參數 OOS 年度為正的比例：", pos.to_dict())

    print("\n[7] 投組模擬（每筆風險 1%、最多 10 檔、單檔上限 20%、可零股）")
    rows = {}
    for variant, g in trades.groupby("variant"):
        eq, taken = simulate_portfolio(panel, g, cfg.trade)
        eq.to_csv(OUTPUT_DIR / f"equity_{variant}_{tag}.csv", header=["equity"], encoding="utf-8-sig")
        rows[variant] = pd.concat([equity_stats(eq), pd.Series({"trades_taken": len(taken)})])
    bench_eq = bench.loc[trades["entry_date"].min():]
    rows["大盤(價格指數)"] = equity_stats(bench_eq / bench_eq.iloc[0] * cfg.trade.initial_capital)
    print(_fmt(pd.DataFrame(rows).T))
    print(f"\n輸出：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
