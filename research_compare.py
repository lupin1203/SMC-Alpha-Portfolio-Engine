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
  volfilter：排除爆量訊號 (訊號日成交量 ÷ 20 日均量 > 1.9) 有沒有幫助，
            有無季線濾網各測一次，對照組為訊號數相同的隨機進場。
  diversify：進場固定為「FVG 無季線 + 排除爆量」，比較持有 5 / 10 / 20 / 30 檔、
            不同單筆風險的組合，並和買進持有 0050 比較。

輸出兩層結果：
  1. 訊號層：每一個訊號都當作獨立交易的平均 R —— 不受資金、持倉上限影響，
     最能直接回答「FVG 進場有沒有比隨機進場好」。
  2. 組合層：經過資金、持倉數、產業上限、市場環境過濾後的真實帳戶績效。
"""
import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

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


# 爆量過濾實驗：門檻 1.9 來自研究期的五分位結果 (最高 20% 約 > 1.89 倍)，
# 所以這個門檻本身是「看過研究期資料」挑的，研究期內的改善會偏樂觀，最後要靠考試期驗證。
VOL_SURGE_CAP = 1.9
# (名稱, 訊號層設定)：一次只改一項
VOLFILTER_SIGNAL = [
    ("A. FVG + 季線（原策略）", dict(REALISTIC)),
    ("B. A + 排除爆量", dict(REALISTIC, max_vol_surge=VOL_SURGE_CAP)),
    ("C. FVG 無季線", dict(REALISTIC, use_sma=False)),
    ("D. C + 排除爆量", dict(REALISTIC, use_sma=False, max_vol_surge=VOL_SURGE_CAP)),
]
VOLFILTER_RANDOM = ("隨機進場（訊號數同 D）", dict(REALISTIC, use_sma=False, max_vol_surge=VOL_SURGE_CAP, random_entry=True))
# (名稱, 在 D 之上再改的組合層設定, 種子參數或 None)
VOLFILTER_PORTFOLIO = [
    ("D + 排序：爆量倍數低優先", dict(rank_by="volume_surge_asc"), None),
    ("D + 風險 0.5%", dict(risk_pct=0.5), None),
    ("D + 排序：隨機", dict(rank_by="random"), "rank_seed"),
]


def _signal_summary(signals):
    if signals is None or signals.empty:
        return {"訊號數": 0, "平均R": np.nan, "標準誤": np.nan, "勝率(%)": np.nan, "R中位數": np.nan, "去掉前5大後平均R": np.nan}
    r = signals['真實R'].astype(float)
    return {
        "訊號數": len(r),
        "平均R": round(r.mean(), 3),
        "標準誤": round(r.std() / np.sqrt(len(r)), 3),
        "勝率(%)": round((r > 0).mean() * 100, 1),
        "R中位數": round(r.median(), 3),
        "去掉前5大後平均R": round(r.sort_values(ascending=False).iloc[5:].mean(), 3) if len(r) > 5 else np.nan,
    }


class Checkpoint:
    """把每一步的結果存成 JSON。整個實驗要跑很久，中途當掉時重跑同一個指令會從上次的進度接著做。"""

    def __init__(self, path, signature):
        self.path = path
        self.store = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("signature") == signature:
                self.store = saved.get("results", {})
                print(f"從進度檔接續：已完成 {len(self.store)} 步 ({path})")
        self.signature = signature

    def get(self, key, compute):
        if key not in self.store:
            self.store[key] = compute()
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"signature": self.signature, "results": self.store}, f, ensure_ascii=False, default=str)
        return self.store[key]


def run_volfilter_suite(args, data, usable, sector_map, sectors):
    last = max(df.index.max() for df in data.values()).date()
    ckpt = Checkpoint(os.path.join(args.out, "_volfilter_progress.json"),
                      f"{len(data)}|{last}|{args.seeds}|{VOL_SURGE_CAP}|{sorted(sectors)}|{BASE_PARAMS}")

    def signal_step(cfg):
        signals = collect_signals(data, sector_map, **cfg)
        by_year = {}
        if not signals.empty:
            by_year = signals.groupby(pd.to_datetime(signals['訊號日']).dt.year)['真實R'].mean().round(2)
            by_year = {str(k): float(v) for k, v in by_year.items()}
        return {"summary": _signal_summary(signals), "by_year": by_year}

    sig_rows, port_rows, yearly = [], [], {}
    for name, cfg in VOLFILTER_SIGNAL:
        print(f"\n▶ {name}")
        s = ckpt.get(f"sig|{name}", lambda: signal_step(cfg))
        sig_rows.append({"設定": name, **s["summary"]})
        yearly[name] = pd.Series({int(k): v for k, v in s["by_year"].items()})
        port_rows.append({"設定": name, **ckpt.get(
            f"port|{name}", lambda: portfolio_stats(usable, sector_map, args.years, **cfg))})

    name, cfg = VOLFILTER_RANDOM
    rand_sig, rand_port = [], []
    for seed in range(args.seeds):
        print(f"\n▶ {name} seed={seed}")
        seed_cfg = dict(cfg, random_seed=seed)
        rand_sig.append(ckpt.get(f"rsig|{seed}", lambda: signal_step(seed_cfg))["summary"])
        rand_port.append(ckpt.get(f"rport|{seed}", lambda: portfolio_stats(usable, sector_map, args.years, **seed_cfg)))
    rs = pd.DataFrame(rand_sig)
    sig_rows.append({"設定": f"{name}，{args.seeds} 組平均", **{c: round(rs[c].mean(), 3) for c in rs.columns}})
    sig_rows.append({"設定": f"{name}，各組平均R的標準差", "平均R": round(rs["平均R"].std(), 3)})
    rp = pd.DataFrame(rand_port).apply(pd.to_numeric, errors="coerce")
    port_rows.append({"設定": f"{name}，{args.seeds} 組平均", **{c: round(rp[c].mean(), 2) for c in rp.columns}})
    port_rows.append({"設定": f"{name}，最差 / 最好報酬(%)",
                      "真實總報酬(%)": f"{rp['真實總報酬(%)'].min():.1f} / {rp['真實總報酬(%)'].max():.1f}"})

    base = VOLFILTER_SIGNAL[-1][1]
    for pname, pcfg, seed_key in VOLFILTER_PORTFOLIO:
        full = dict(base, **pcfg)
        if seed_key is None:
            print(f"\n▶ {pname}")
            port_rows.append({"設定": pname, **ckpt.get(
                f"port|{pname}", lambda: portfolio_stats(usable, sector_map, args.years, **full))})
            continue
        res = []
        for seed in range(args.seeds):
            print(f"\n▶ {pname} seed={seed}")
            seeded = dict(full, **{seed_key: seed})
            res.append(ckpt.get(f"port|{pname}|{seed}",
                                lambda: portfolio_stats(usable, sector_map, args.years, **seeded)))
        rk = pd.DataFrame(res).apply(pd.to_numeric, errors="coerce")
        port_rows.append({"設定": f"{pname}，{args.seeds} 組平均", **{c: round(rk[c].mean(), 2) for c in rk.columns}})
        port_rows.append({"設定": f"{pname}，最差 / 最好報酬(%)",
                          "真實總報酬(%)": f"{rk['真實總報酬(%)'].min():.1f} / {rk['真實總報酬(%)'].max():.1f}"})

    sig_df, port_df = pd.DataFrame(sig_rows), pd.DataFrame(port_rows)
    year_df = pd.DataFrame(yearly)
    year_df.index.name = "訊號年份"
    d_r = sig_rows[len(VOLFILTER_SIGNAL) - 1]["平均R"]
    beaten = (rs["平均R"] < d_r).mean() * 100
    verdict = (f"D 的平均 R ({d_r}) 贏過 {beaten:.0f}% 的隨機進場對照組"
               f"（對照組平均 {rs['平均R'].mean():.3f}，標準差 {rs['平均R'].std():.3f}）。")

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(args.out, f"compare_volfilter_{stamp}.md")
    first = min(df.index.min() for df in data.values()).date()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 爆量過濾實驗結果 ({stamp})\n\n")
        f.write(f"- 研究期：{first} ~ {HOLDOUT_START} 之前（考試期未使用）\n")
        f.write(f"- 板塊：{'、'.join(sectors)}；有效標的 {len(data)} 檔\n")
        f.write(f"- 爆量上限：訊號日成交量 ÷ 20 日均量 ≤ {VOL_SURGE_CAP}（門檻是看研究期資料挑的，研究期內的改善會偏樂觀）\n")
        f.write(f"- 全部使用實盤真實模式；基本參數：{BASE_PARAMS}\n\n")
        f.write("## 訊號層（每個訊號獨立計算，不受資金限制）\n\n" + sig_df.to_markdown(index=False) + "\n\n")
        f.write(f"- **{verdict}**\n\n")
        f.write("### 各年度訊號平均 R\n\n" + year_df.to_markdown() + "\n\n")
        f.write("## 組合層（資金、持倉、產業上限、市場環境過濾後）\n\n" + port_df.to_markdown(index=False) + "\n")

    print("\n" + "=" * 60)
    print("訊號層：\n" + sig_df.to_string(index=False))
    print("\n各年度訊號平均 R：\n" + year_df.to_string())
    print("\n組合層：\n" + port_df.to_string(index=False))
    print("\n" + verdict)
    print(f"\n結果已存到 {path}")


# 分散持有實驗：進場固定為 D（FVG 無季線 + 排除爆量），只改組合層的持倉數與風險。
# 想法：訊號層有小幅優勢，但 5 檔組合只買到不到 1% 的訊號，結果主要看運氣；
# 持有更多檔、每筆風險更小，帳戶報酬應該更接近訊號層的平均。
DIVERSIFY_ENTRY = dict(REALISTIC, use_sma=False, max_vol_surge=VOL_SURGE_CAP)
# (名稱, 組合層設定)；max_position_pct = 單筆最多佔淨值多少，避免現金被少數幾檔吃光
DIVERSIFY_PORTFOLIO = [
    ("5 檔、風險 1.5%（目前設定）", dict(max_concurrent=5, max_per_sector=2, risk_pct=1.5, max_position_pct=0.20)),
    ("10 檔、風險 0.5%", dict(max_concurrent=10, max_per_sector=3, risk_pct=0.5, max_position_pct=0.10)),
    ("20 檔、風險 0.3%", dict(max_concurrent=20, max_per_sector=5, risk_pct=0.3, max_position_pct=0.05)),
    ("30 檔、風險 0.2%", dict(max_concurrent=30, max_per_sector=6, risk_pct=0.2, max_position_pct=0.033)),
    ("30 檔、風險 0.2%、不過濾市場環境",
     dict(max_concurrent=30, max_per_sector=6, risk_pct=0.2, max_position_pct=0.033, allowed_regimes=None)),
]
# 這兩組再用隨機排序跑幾次，看結果受「同一天挑哪幾檔」的運氣影響多大
DIVERSIFY_RANK_CHECK = ["20 檔、風險 0.3%", "30 檔、風險 0.2%"]


def _buy_hold(ticker, years, start, end):
    df = fetch_stock_data(ticker, days=years * 365)
    if df is None or df.empty:
        return None
    c = df['Close'][(df.index >= start) & (df.index < end)]
    if len(c) < 2:
        return None
    peak = c.cummax()
    return {"真實總報酬(%)": round((c.iloc[-1] / c.iloc[0] - 1) * 100, 2),
            "逐日盯市最大回撤(%)": round(((peak - c) / peak).max() * 100, 2)}


def run_diversify_suite(args, data, usable, sector_map, sectors):
    holdout_ts = pd.Timestamp(HOLDOUT_START)
    first = min(df.index.min() for df in data.values())
    last = max(df.index.max() for df in data.values())
    span_years = (last - first).days / 365.25
    ckpt = Checkpoint(os.path.join(args.out, "_diversify_progress.json"),
                      f"{len(data)}|{last.date()}|{args.seeds}|{VOL_SURGE_CAP}|{sorted(sectors)}|{BASE_PARAMS}")

    def run(cfg):
        return portfolio_stats(usable, sector_map, args.years, **dict(DIVERSIFY_ENTRY, **cfg))

    rows = []
    for name, cfg in DIVERSIFY_PORTFOLIO:
        print(f"\n▶ {name}")
        rows.append({"設定": name, **ckpt.get(f"port|{name}", lambda: run(cfg))})

    for name in DIVERSIFY_RANK_CHECK:
        cfg = dict(DIVERSIFY_PORTFOLIO)[name]
        res = []
        for seed in range(args.seeds):
            print(f"\n▶ {name}，隨機排序 seed={seed}")
            seeded = dict(cfg, rank_by="random", rank_seed=seed)
            res.append(ckpt.get(f"rank|{name}|{seed}", lambda: run(seeded)))
        rk = pd.DataFrame(res).apply(pd.to_numeric, errors="coerce")
        rows.append({"設定": f"{name}，隨機排序 {args.seeds} 組平均", **{c: round(rk[c].mean(), 2) for c in rk.columns}})
        rows.append({"設定": f"{name}，隨機排序 最差 / 最好報酬(%)",
                     "真實總報酬(%)": f"{rk['真實總報酬(%)'].min():.1f} / {rk['真實總報酬(%)'].max():.1f}"})

    # 對照：同樣 30 檔分散，但進場日期隨機 (訊號數相同)
    rnd_cfg = dict(DIVERSIFY_PORTFOLIO)["30 檔、風險 0.2%"]
    res = []
    for seed in range(args.seeds):
        print(f"\n▶ 隨機進場，30 檔 seed={seed}")
        seeded = dict(rnd_cfg, random_entry=True, random_seed=seed)
        res.append(ckpt.get(f"rand30|{seed}", lambda: run(seeded)))
    rk = pd.DataFrame(res).apply(pd.to_numeric, errors="coerce")
    rows.append({"設定": f"隨機進場，30 檔、風險 0.2%，{args.seeds} 組平均", **{c: round(rk[c].mean(), 2) for c in rk.columns}})
    rows.append({"設定": "隨機進場，30 檔 最差 / 最好報酬(%)",
                 "真實總報酬(%)": f"{rk['真實總報酬(%)'].min():.1f} / {rk['真實總報酬(%)'].max():.1f}"})

    bh = _buy_hold("0050.TW", args.years, first, holdout_ts)
    if bh:
        rows.append({"設定": "買進持有 0050（同期間）", **bh})

    df = pd.DataFrame(rows)
    ret = pd.to_numeric(df["真實總報酬(%)"], errors="coerce")
    df.insert(df.columns.get_loc("真實總報酬(%)") + 1, "年化報酬(%)",
              (((1 + ret / 100) ** (1 / span_years) - 1) * 100).round(2))

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(args.out, f"compare_diversify_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 分散持有實驗結果 ({stamp})\n\n")
        f.write(f"- 研究期：{first.date()} ~ {HOLDOUT_START} 之前（約 {span_years:.1f} 年，考試期未使用）\n")
        f.write(f"- 板塊：{'、'.join(sectors)}；有效標的 {len(data)} 檔\n")
        f.write(f"- 進場固定：FVG 無季線 + 排除爆量 (> {VOL_SURGE_CAP} 倍)，實盤真實模式\n")
        f.write("- 只改組合層：同時持倉數、單筆風險、單筆資金上限、同產業上限\n")
        f.write(f"- 其他參數：{BASE_PARAMS}\n\n")
        f.write(df.to_markdown(index=False) + "\n")
    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print(f"\n結果已存到 {path}")


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
    # Windows 上多進程偶爾會有子進程意外結束 (BrokenProcessPool)，重試即可，不必整個實驗重來
    for attempt in range(3):
        try:
            _, stats, _ = run_portfolio_backtest(
                tickers=tickers, years=years, sector_map=sector_map, period="in_sample", **{**BASE_PARAMS, **cfg})
            break
        except BrokenProcessPool:
            if attempt == 2:
                raise
            print(f"⚠️ 子進程意外結束，重試第 {attempt + 2} 次")
    if not stats:
        return {}
    keep = ["總交易次數", "勝率(%)", "Portfolio期望值(R)", "真實總報酬(%)", "真實最大回撤(%)",
            "逐日盯市最大回撤(%)", "夏普值(Sharpe)"]
    return {k: stats.get(k) for k in keep}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sectors", nargs="*", help="要測試的板塊 (預設全部)")
    ap.add_argument("--years", type=int, default=12, help="抓取的資料年數")
    ap.add_argument("--seeds", type=int, default=10, help="隨機對照組的種子數")
    ap.add_argument("--out", default="research_results", help="結果輸出資料夾")
    ap.add_argument("--suite", default="baseline", choices=list(SUITES) + ["portfolio", "volfilter", "diversify"], help="實驗組")
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
    if args.suite == "volfilter":
        run_volfilter_suite(args, data, usable, sector_map, sectors)
        return
    if args.suite == "diversify":
        run_diversify_suite(args, data, usable, sector_map, sectors)
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
