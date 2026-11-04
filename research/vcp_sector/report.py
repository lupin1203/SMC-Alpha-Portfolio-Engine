"""Phase 9（先做 Phase 3 需要的部分）：族群強度 CSV + 熱力圖。

用法（在 repo 根目錄）::

    python -m research.vcp_sector.report                  # 官方產業別，真實資料
    python -m research.vcp_sector.report --groups theme   # groups.yaml 題材
    python -m research.vcp_sector.report --synthetic      # 人造資料，檢查流程用
    python -m research.vcp_sector.report --no-api         # 只讀快取，不打 FinMind

輸出到 ``research/vcp_sector/output/``：
- ``sector_strength_<groups>.csv``：每日 × 族群全部指標
- ``sector_heatmap_<groups>.html``：族群 × 日期熱力圖（顏色 = 指標，★ = is_strong）
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from .config import DEFAULT_CONFIG, MODULE_DIR
from .data_loader import (eligibility_mask, load_benchmark, load_prices, load_stock_info,
                          to_wide)
from .sector_groups import official_groups, theme_groups
from .sector_strength import compute_sector_strength

OUTPUT_DIR = MODULE_DIR / "output"
METRIC_LABELS = {
    "group_rank": "族群 RS 百分位排名",
    "pct_above_ma20": "站上 MA20 比例",
    "pct_above_ma50": "站上 MA50 比例",
    "pct_new_high_20": "創 20 日新高比例",
    "group_rs": "族群 20 日相對大盤報酬",
}


def plot_strength_heatmap(strength: pd.DataFrame, metric: str = "group_rank",
                          title: str | None = None, last_n_days: int | None = None):
    """族群 × 日期熱力圖。is_strong 的格子用 ★ 標記。回傳 plotly Figure。"""
    import plotly.graph_objects as go

    df = strength.copy()
    if last_n_days:
        keep = sorted(df["date"].unique())[-last_n_days:]
        df = df[df["date"].isin(keep)]
    z = df.pivot(index="group", columns="date", values=metric)
    # 去掉暖機期（全 NaN 的日期）與從未達成員門檻的族群
    z = z.dropna(axis=1, how="all").dropna(axis=0, how="all")
    strong = df.pivot(index="group", columns="date", values="is_strong").reindex_like(z)
    members = df.pivot(index="group", columns="date", values="n_members").reindex_like(z)

    # 依最近一日的指標排序，強的在上
    order = z.iloc[:, -1].fillna(-1e9).sort_values().index
    z, strong, members = z.loc[order], strong.loc[order], members.loc[order]

    pct = metric != "group_rs"
    fig = go.Figure(go.Heatmap(
        z=z.values, x=z.columns, y=z.index,
        colorscale="RdYlGn", zmin=0 if pct else None, zmax=1 if pct else None,
        zmid=None if pct else 0,
        colorbar=dict(title=METRIC_LABELS.get(metric, metric), tickformat=".0%"),
        customdata=members.values,
        hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>" + metric + "=%{z:.2%}"
                      "<br>有效成員=%{customdata}<extra></extra>",
    ))
    ys, xs = (strong.fillna(False).astype(bool).values).nonzero()
    if len(xs):
        fig.add_trace(go.Scatter(
            x=z.columns[xs], y=z.index[ys], mode="markers",
            marker=dict(symbol="star", size=6, color="black"),
            name="is_strong", hoverinfo="skip",
        ))
    fig.update_layout(
        title=title or f"族群強度熱力圖：{METRIC_LABELS.get(metric, metric)}（★ = 族群轉強條件成立）",
        height=max(360, 26 * len(z) + 160), xaxis_title="日期", yaxis_title="族群",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=140, r=40, t=80, b=60),
    )
    return fig


def build_strength(groups_mode: str = "official", synthetic: bool = False, use_api: bool = True,
                   cfg=DEFAULT_CONFIG) -> pd.DataFrame:
    if synthetic:
        from .synthetic import make_synthetic_market

        prices, info, bench = make_synthetic_market()
    else:
        info = load_stock_info(cfg.data, use_api=use_api)
        prices = load_prices(info["stock_id"], cfg.data, use_api=use_api)
        if prices.empty:
            raise SystemExit("沒有任何價格資料：請先跑 update_data.py 建 local_data/，"
                             "或設定 FINMIND_TOKEN，或用 --synthetic 檢查流程。")
        bench = load_benchmark(cfg.data, use_api=use_api)

    universe = set(prices["stock_id"].unique())
    if groups_mode == "theme":
        groups = theme_groups(cfg.sector.groups_yaml, universe)
    else:
        groups = official_groups(info[info["stock_id"].isin(universe)])

    close = to_wide(prices, "close")
    elig = eligibility_mask(prices, cfg.data)
    return compute_sector_strength(close, groups, bench, elig, cfg.sector)


def main(argv=None):
    p = argparse.ArgumentParser(description="族群強度（Phase 1–3）")
    p.add_argument("--groups", choices=["official", "theme"], default="official")
    p.add_argument("--synthetic", action="store_true", help="用人造資料（非市場資料）")
    p.add_argument("--no-api", action="store_true", help="只讀快取，不呼叫 FinMind")
    p.add_argument("--metric", default="group_rank", choices=list(METRIC_LABELS))
    p.add_argument("--last-n-days", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    strength = build_strength(args.groups, args.synthetic, not args.no_api)
    tag = ("synthetic_" if args.synthetic else "") + args.groups
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"sector_strength_{tag}.csv"
    html_path = OUTPUT_DIR / f"sector_heatmap_{tag}.html"
    strength.to_csv(csv_path, index=False, encoding="utf-8-sig")

    title = None
    if args.synthetic:
        title = f"【人造資料】族群強度熱力圖：{METRIC_LABELS[args.metric]}（★ = 轉強）"
    fig = plot_strength_heatmap(strength, args.metric, title, args.last_n_days)
    fig.write_html(html_path, include_plotlyjs="cdn")

    last = strength[strength["date"] == strength["date"].max()]
    print(f"資料期間：{strength['date'].min():%Y-%m-%d} ~ {strength['date'].max():%Y-%m-%d}，"
          f"族群數 {strength['group'].nunique()}")
    print(f"最新一日轉強族群：{last.loc[last['is_strong'], 'group'].tolist()}")
    print(f"CSV  → {csv_path}\nHTML → {html_path}")


if __name__ == "__main__":
    main()
