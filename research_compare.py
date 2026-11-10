# research_compare.py
"""
對照實驗：原始回測 vs 實盤真實模式 vs 隨機進場對照組。

只使用研究期資料 (config.HOLDOUT_START 之前)，考試期完全不碰。

用法：
    python research_compare.py                       # 全部板塊，baseline 實驗組
    python research_compare.py --suite entry         # 進場方式拆解實驗
    python research_compare.py --sectors 半導體業 電子零組件
    python research_compare.py --seeds 20            # 隨機對照組跑 20 組種子

實驗組 (--suite)：
  baseline：原始回測 / 實盤真實模式 / 隨機進場 —— FVG 進場有沒有比隨機好？
  entry   ：全部用實盤真實模式，比較 FVG / 突破 20 日新高 / 只在季線上隨機進場 / 任意日隨機進場
            —— 拆解是哪個進場條件在加分或扣分。

輸出兩層結果：
  1. 訊號層：每一個訊號都當作獨立交易的平均 R —— 不受資金、持倉上限影響，
     最能直接回答「FVG 進場有沒有比隨機進場好」。
  2. 組合層：經過資金、持倉數、產業上限、市場環境過濾後的真實帳戶績效。
"""
import argparse
import datetime
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from config import HOLDOUT_START
from data_provider import load_data, fetch_stock_data
from portfolio_engine import run_portfolio_backtest, _pure_calc_worker

# README 中的「目前驗證過的最佳參數」(回測版)
BASE_PARAMS = dict(
    bt_mode="移動停利", bt_val=10.0, gap_pct=0.3, use_sma=True,
    slippage_pct=0.002, fee_disc=0.28, min_vol=1000,
    max_concurrent=5, max_per_sector=2, risk_pct=1.5,
    initial_capital=1000000.0, account_dd_limit=0.20,
)

REALISTIC = dict(realistic=True, max_position_pct=0.20)

# 每個實驗：(名稱, 設定, 是否為隨機對照組)
SUITES = {
    "baseline": [
        ("原始回測", dict(realistic=False), False),
        ("實盤真實模式", REALISTIC, False),
        ("隨機進場 (實盤模式)", dict(REALISTIC, random_entry=True), True),
    ],
    "entry": [
        ("FVG + 季線", REALISTIC, False),
        ("突破20日新高 + 季線", dict(REALISTIC, entry_type="breakout20"), False),
        ("隨機進場 (只在季線上)", dict(REALISTIC, random_entry=True, random_within_sma=True), True),
        ("隨機進場 (任意日)", dict(REALISTIC, random_entry=True), True),
    ],
}


def signal_level_stats(data, sector_map, **cfg):
    """每個訊號獨立計算 R (不經過組合層篩選)。"""
    worker_kwargs = {k: v for k, v in {**BASE_PARAMS, **cfg}.items()
                     if k not in ('max_concurrent', 'max_per_sector', 'risk_pct', 'initial_capital',
                                  'account_dd_limit', 'max_position_pct')}
    frames = []
    for t, df in data.items():
        res = _pure_calc_worker(t, df.copy(), sector_map=sector_map, **worker_kwargs)
        if res is not None and not res.empty:
            frames.append(res)
    if not frames:
        return {"訊號數": 0, "平均R": np.nan, "勝率(%)": np.nan, "R中位數": np.nan, "去掉前5大後平均R": np.nan}
    r = pd.concat(frames)['真實R'].astype(float)
    top5_removed = r.sort_values(ascending=False).iloc[5:]
    return {
        "訊號數": len(r),
        "平均R": round(r.mean(), 3),
        "勝率(%)": round((r > 0).mean() * 100, 1),
        "R中位數": round(r.median(), 3),
        "去掉前5大後平均R": round(top5_removed.mean(), 3) if len(top5_removed) else np.nan,
    }


def portfolio_stats(tickers, sector_map, years, **cfg):
    _, stats, _ = run_portfolio_backtest(
        tickers=tickers, years=years, sector_map=sector_map, period="in_sample", **{**BASE_PARAMS, **cfg})
    if not stats:
        return {}
    keep = ["總交易次數", "勝率(%)", "Portfolio期望值(R)", "真實總報酬(%)", "真實最大回撤(%)", "夏普值(Sharpe)"]
    return {k: stats.get(k) for k in keep}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sectors", nargs="*", help="要測試的板塊 (預設全部)")
    ap.add_argument("--years", type=int, default=12, help="抓取的資料年數")
    ap.add_argument("--seeds", type=int, default=10, help="隨機對照組的種子數")
    ap.add_argument("--out", default="research_results", help="結果輸出資料夾")
    ap.add_argument("--suite", default="baseline", choices=list(SUITES), help="實驗組")
    args = ap.parse_args()

    industry, _, _ = load_data()
    sectors = args.sectors or list(industry.keys())
    unknown = [s for s in sectors if s not in industry]
    if unknown:
        sys.exit(f"找不到板塊：{unknown}。可用板塊：{list(industry.keys())}")
    tickers = sorted({t for s in sectors for t in industry[s]})
    sector_map = {t: s for s, ts in industry.items() for t in ts}
    print(f"板塊 {len(sectors)} 個、標的 {len(tickers)} 檔，研究期 < {HOLDOUT_START}")

    # 先把資料抓齊 (之後的實驗都讀本地快取)
    holdout_ts = pd.Timestamp(HOLDOUT_START)

    def _load(t):
        df = fetch_stock_data(t, days=args.years * 365)
        if df is None or df.empty:
            return t, None
        df = df[df.index < holdout_ts]
        return t, (df if len(df) >= 100 else None)

    with ThreadPoolExecutor(max_workers=8) as ex:
        data = {t: df for t, df in ex.map(_load, tickers) if df is not None}
    print(f"有效資料 {len(data)} 檔")
    if not data:
        sys.exit("沒有任何可用資料，請先執行 update_data.py")
    usable = list(data.keys())

    sig_rows, port_rows = [], []
    random_groups = {}   # 名稱 -> 各種子的訊號層平均 R
    deterministic = {}   # 名稱 -> 訊號層平均 R
    for name, cfg, is_random in SUITES[args.suite]:
        if not is_random:
            print(f"\n▶ {name}")
            sig = signal_level_stats(data, sector_map, **cfg)
            deterministic[name] = sig["平均R"]
            sig_rows.append({"設定": name, **sig})
            port_rows.append({"設定": name, **portfolio_stats(usable, sector_map, args.years, **cfg)})
            continue

        rand_sig, rand_port = [], []
        for seed in range(args.seeds):
            print(f"\n▶ {name} seed={seed}")
            seed_cfg = dict(cfg, random_seed=seed)
            rand_sig.append(signal_level_stats(data, sector_map, **seed_cfg))
            rand_port.append(portfolio_stats(usable, sector_map, args.years, **seed_cfg))
        rs = pd.DataFrame(rand_sig)
        random_groups[name] = rs["平均R"]
        sig_rows.append({"設定": f"{name}，{args.seeds} 組平均", **{c: round(rs[c].mean(), 3) for c in rs.columns}})
        sig_rows.append({"設定": f"{name}，標準差", **{c: round(rs[c].std(), 3) for c in rs.columns}})
        rp = pd.DataFrame(rand_port).apply(pd.to_numeric, errors="coerce")
        if not rp.empty:
            port_rows.append({"設定": f"{name}，{args.seeds} 組平均", **{c: round(rp[c].mean(), 2) for c in rp.columns}})
            port_rows.append({"設定": f"{name}，最差 / 最好報酬(%)",
                              "真實總報酬(%)": f"{rp['真實總報酬(%)'].min():.1f} / {rp['真實總報酬(%)'].max():.1f}"})

    sig_df = pd.DataFrame(sig_rows)
    port_df = pd.DataFrame(port_rows)

    verdicts = []
    for d_name, d_r in deterministic.items():
        for r_name, r_series in random_groups.items():
            beaten = (r_series < d_r).mean() * 100
            verdicts.append(f"{d_name} 的平均 R ({d_r}) 贏過 {beaten:.0f}% 的「{r_name}」對照組"
                            f"（對照組平均 {r_series.mean():.3f}）。")
    verdict = "\n".join(verdicts)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(args.out, f"compare_{args.suite}_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 對照實驗結果：{args.suite} ({stamp})\n\n")
        f.write(f"- 研究期：< {HOLDOUT_START}（考試期未使用）\n")
        f.write(f"- 板塊：{'、'.join(sectors)}；有效標的 {len(data)} 檔\n")
        f.write(f"- 參數：{BASE_PARAMS}\n\n")
        f.write("## 訊號層（每個訊號獨立計算，不受資金限制）\n\n")
        f.write(sig_df.to_markdown(index=False) + "\n\n")
        for v in verdicts:
            f.write(f"- **{v}**\n")
        f.write("\n")
        f.write("## 組合層（資金、持倉、產業上限、市場環境過濾後）\n\n")
        f.write(port_df.to_markdown(index=False) + "\n")

    print("\n" + "=" * 60)
    print("訊號層：")
    print(sig_df.to_string(index=False))
    print("\n組合層：")
    print(port_df.to_string(index=False))
    if verdict:
        print("\n" + verdict)
    print(f"\n結果已存到 {path}")


if __name__ == "__main__":
    main()
