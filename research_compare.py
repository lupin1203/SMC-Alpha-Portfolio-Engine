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
  pullback：FVG 有無季線濾網 × 追價進場 / 回檔到缺口上緣才進場，
            對照組為任意日隨機進場 (追價 / 回檔到當日低點)。
  portfolio：進場固定為「FVG 無季線、追價」，研究組合層：
            (1) 訊號層拆解：依市場環境、爆量倍數五分位看平均 R
            (2) 組合回測：排序方式、市場環境過濾、單筆風險，一次只改一項

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
    "pullback": [
        ("FVG + 季線，追價", REALISTIC, False),
        ("FVG 無季線，追價", dict(REALISTIC, use_sma=False), False),
        ("FVG + 季線，回檔進場", dict(REALISTIC, entry_mode="retest"), False),
        ("FVG 無季線，回檔進場", dict(REALISTIC, use_sma=False, entry_mode="retest"), False),
        ("隨機進場，追價", dict(REALISTIC, random_entry=True), True),
        ("隨機進場，回檔進場", dict(REALISTIC, random_entry=True, entry_mode="retest"), True),
    ],
}


# 組合層實驗：進場固定為目前訊號層最好的設定
PORTFOLIO_ENTRY = dict(REALISTIC, use_sma=False)
# (名稱, 組合層設定, 要變動的種子參數或 None)
PORTFOLIO_SUITE = [
    ("基準：爆量優先、Bull_LowVol+Bear_HighVol、風險1.5%", dict(), None),
    ("排序：爆量倍數低優先", dict(rank_by="volume_surge_asc"), None),
    ("排序：隨機", dict(rank_by="random"), "rank_seed"),
    ("環境：不過濾", dict(allowed_regimes=None), None),
    ("環境：只做 Bull", dict(allowed_regimes=("Bull_LowVol", "Bull_HighVol")), None),
    ("風險：0.5%", dict(risk_pct=0.5), None),
]


def collect_signals(data, sector_map, **cfg):
    worker_kwargs = {k: v for k, v in {**BASE_PARAMS, **cfg}.items()
                     if k not in ('max_concurrent', 'max_per_sector', 'risk_pct', 'initial_capital',
                                  'account_dd_limit', 'max_position_pct', 'rank_by', 'rank_seed',
                                  'allowed_regimes')}
    frames = []
    for t, df in data.items():
        res = _pure_calc_worker(t, df.copy(), sector_map=sector_map, **worker_kwargs)
        if res is not None and not res.empty:
            frames.append(res)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def signal_breakdown(signals, market_df):
    """依市場環境、爆量倍數五分位拆解訊號平均 R。"""
    from regime import add_regime_columns
    out = {}
    sig_dates = pd.to_datetime(signals['訊號日'])
    m = add_regime_columns(market_df)
    m = m[~m.index.duplicated(keep='last')].sort_index()
    pos = m.index.get_indexer(sig_dates, method='pad')
    regime = np.where(pos >= 0, m['Regime'].values[np.clip(pos, 0, None)], 'Unknown')
    df = pd.DataFrame({'R': signals['真實R'].astype(float).values, '環境': regime,
                       '爆量倍數': signals['Alpha動能分數'].astype(float).values})

    def _agg(g):
        return pd.Series({"訊號數": len(g), "平均R": round(g['R'].mean(), 3),
                          "標準誤": round(g['R'].std() / np.sqrt(len(g)), 3),
                          "勝率(%)": round((g['R'] > 0).mean() * 100, 1)})

    out['市場環境'] = df.groupby('環境').apply(_agg, include_groups=False).reset_index()
    valid = df.dropna(subset=['爆量倍數'])
    valid = valid.assign(五分位=pd.qcut(valid['爆量倍數'], 5, labels=["Q1 最低", "Q2", "Q3", "Q4", "Q5 最高"]))
    q = valid.groupby('五分位', observed=True).apply(_agg, include_groups=False).reset_index()
    rng = valid.groupby('五分位', observed=True)['爆量倍數'].agg(['min', 'max']).round(2).reset_index(drop=True)
    q.insert(1, '爆量倍數範圍', rng['min'].astype(str) + " – " + rng['max'].astype(str))
    out['爆量倍數五分位'] = q
    return out


def run_portfolio_suite(args, data, usable, sector_map, sectors):
    holdout_ts = pd.Timestamp(HOLDOUT_START)
    market = fetch_stock_data("0050.TW", days=args.years * 365)
    market = market[market.index < holdout_ts]

    print("\n▶ 訊號層拆解")
    signals = collect_signals(data, sector_map, **PORTFOLIO_ENTRY)
    breakdown = signal_breakdown(signals, market)

    rows = []
    for name, cfg, seed_key in PORTFOLIO_SUITE:
        full = dict(PORTFOLIO_ENTRY, **cfg)
        if seed_key is None:
            print(f"\n▶ {name}")
            rows.append({"設定": name, **portfolio_stats(usable, sector_map, args.years, **full)})
            continue
        res = []
        for seed in range(args.seeds):
            print(f"\n▶ {name} seed={seed}")
            res.append(portfolio_stats(usable, sector_map, args.years, **dict(full, **{seed_key: seed})))
        rp = pd.DataFrame(res).apply(pd.to_numeric, errors="coerce")
        rows.append({"設定": f"{name}，{args.seeds} 組平均", **{c: round(rp[c].mean(), 2) for c in rp.columns}})
        rows.append({"設定": f"{name}，最差 / 最好報酬(%)",
                     "真實總報酬(%)": f"{rp['真實總報酬(%)'].min():.1f} / {rp['真實總報酬(%)'].max():.1f}"})
    port_df = pd.DataFrame(rows)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(args.out, f"compare_portfolio_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 組合層實驗結果 ({stamp})\n\n")
        f.write(f"- 研究期：< {HOLDOUT_START}（考試期未使用）\n")
        f.write(f"- 板塊：{'、'.join(sectors)}；有效標的 {len(data)} 檔\n")
        f.write(f"- 進場：FVG 無季線、追價（實盤真實模式）\n- 基本參數：{BASE_PARAMS}\n\n")
        f.write(f"## 訊號層拆解（共 {len(signals)} 個訊號）\n\n")
        for title, table in breakdown.items():
            f.write(f"### 依{title}\n\n{table.to_markdown(index=False)}\n\n")
        f.write("## 組合回測（一次只改一項）\n\n")
        f.write(port_df.to_markdown(index=False) + "\n")

    print("\n" + "=" * 60)
    for title, table in breakdown.items():
        print(f"\n依{title}：")
        print(table.to_string(index=False))
    print("\n組合回測：")
    print(port_df.to_string(index=False))
    print(f"\n結果已存到 {path}")


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
    ap.add_argument("--suite", default="baseline", choices=list(SUITES) + ["portfolio"], help="實驗組")
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

    if args.suite == "portfolio":
        run_portfolio_suite(args, data, usable, sector_map, sectors)
        return

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
