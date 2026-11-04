"""VCP 人工檢查頁：抽樣突破案例，K 線上標註 swing 點、確認日、pivot、停損、突破日。

用法：python -m research.vcp_sector.vcp_review [--n 30] [--seed 0] [--stock 2330]
輸出：research/vcp_sector/output/vcp_review.html（瀏覽器打開，可縮放）

圖例：
- 橘線＝ZigZag 連線（只畫突破日當時「已確認」的轉折點）
- ▲▼＝轉折點（畫在轉折實際發生日）；灰色 ×＝該轉折的「確認日」（回測只能從這天起使用）
- 藍虛線＝pivot（最後一段收縮的高點）；紅虛線＝停損（最後一段收縮的低點）
- 綠色 ★＝突破日；灰底＝整理區間（base_start → 突破日）
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG
from .lead_lag import StockPanel
from .swing_points import zigzag


def plot_case(panel: StockPanel, row, cfg=DEFAULT_CONFIG.vcp, pad_before: int = 60, pad_after: int = 30):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    s = row.stock_id
    t = panel.row[row.date]
    b = panel.row[row.base_start]
    lo_i, hi_i = max(0, b - pad_before), min(len(panel.dates) - 1, t + pad_after)
    d = panel.dates[lo_i:hi_i + 1]
    o, h, l, c, v = (getattr(panel, f)[s].iloc[lo_i:hi_i + 1] for f in ("open", "high", "low", "close", "volume"))
    ma50 = panel.close[s].rolling(50, min_periods=50).mean().iloc[lo_i:hi_i + 1]

    sw = zigzag(panel.high[s].to_numpy(float), panel.low[s].to_numpy(float), cfg.min_swing_pct)
    known = [x for x in sw if x[3] <= t and lo_i <= x[0] <= hi_i]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d, open=o, high=h, low=l, close=c, name="K",
                                 increasing_line_color="#d62728", decreasing_line_color="#2ca02c"), 1, 1)
    fig.add_trace(go.Scatter(x=d, y=ma50, name="MA50", line=dict(width=1, color="#7f7f7f")), 1, 1)
    if known:
        fig.add_trace(go.Scatter(x=[panel.dates[x[0]] for x in known], y=[x[1] for x in known],
                                 mode="lines+markers", name="ZigZag（已確認）", line=dict(color="orange", width=1.5),
                                 marker=dict(symbol=["triangle-up" if x[2] == "H" else "triangle-down" for x in known],
                                             size=9, color="orange")), 1, 1)
        fig.add_trace(go.Scatter(x=[panel.dates[x[3]] for x in known], y=[x[1] for x in known], mode="markers",
                                 name="確認日", marker=dict(symbol="x", size=7, color="gray")), 1, 1)
    for y, color, name in ((row.pivot, "#1f77b4", "pivot"), (row.stop_price, "#d62728", "停損")):
        fig.add_trace(go.Scatter(x=[panel.dates[b], d[-1]], y=[y, y], mode="lines", name=name,
                                 line=dict(color=color, dash="dash", width=1)), 1, 1)
    is_bo = bool(getattr(row, "is_breakout", True))
    fig.add_trace(go.Scatter(x=[row.date], y=[c.loc[row.date]], mode="markers",
                             name="突破日" if is_bo else "今日（整理中）",
                             marker=dict(symbol="star" if is_bo else "circle", size=14,
                                         color="green" if is_bo else "purple")), 1, 1)
    chase = row.pivot * (1 + cfg.max_chase_pct)
    fig.add_trace(go.Scatter(x=[panel.dates[b], d[-1]], y=[chase, chase], mode="lines",
                             name=f"追價上限 +{cfg.max_chase_pct:.0%}",
                             line=dict(color="#1f77b4", dash="dot", width=1)), 1, 1)
    fig.add_vrect(x0=panel.dates[b], x1=row.date, fillcolor="gray", opacity=0.08, line_width=0, row=1, col=1)
    fig.add_trace(go.Bar(x=d, y=v, name="量", marker_color="#9ecae1", showlegend=False), 2, 1)
    title = (f"{s}｜突破 {row.date:%Y-%m-%d}｜收縮 {row.n_contractions} 段：{row.contraction_depths}"
             f"｜pivot {row.pivot:.2f}／停損 {row.stop_price:.2f}（{1 - row.stop_price / row.pivot:.1%}）")
    if "r" in row._fields and pd.notna(getattr(row, "r", np.nan)):
        title += f"｜結果 {row.r:+.2f}R（{row.exit_reason}）"
    fig.update_layout(title=dict(text=title, font=dict(size=13)), height=520, xaxis_rangeslider_visible=False,
                      margin=dict(l=50, r=20, t=50, b=30), legend=dict(orientation="h", y=-0.08))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def main(argv=None):
    from .config import DEFAULT_CONFIG as cfg
    from .report import OUTPUT_DIR, load_inputs
    from .vcp_detector import detect_all

    p = argparse.ArgumentParser(description="VCP 人工檢查頁")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stock", default=None, help="只看某一檔")
    p.add_argument("--no-api", action="store_true")
    args = p.parse_args(argv)

    csv = OUTPUT_DIR / "vcp_signals_official.csv"
    inp = load_inputs("official", False, not args.no_api, cfg)
    panel = StockPanel(inp["prices"], inp["eligible"], cfg.leadlag, cfg.data.turnover_window)
    if csv.exists():
        vcp = pd.read_csv(csv, parse_dates=["date", "base_start", "last_low_date"], dtype={"stock_id": str})
    else:
        vcp = detect_all(panel, cfg.vcp)
    bo = vcp[vcp["is_breakout"]].copy()
    trades_csv = OUTPUT_DIR / "trades_vcp_official.csv"
    if trades_csv.exists():
        tr = pd.read_csv(trades_csv, parse_dates=["signal_date"], dtype={"stock_id": str})
        tr = tr[tr["variant"] == "A_vcp"][["signal_date", "stock_id", "r", "exit_reason"]]
        bo = bo.merge(tr, left_on=["date", "stock_id"], right_on=["signal_date", "stock_id"], how="left")
    if args.stock:
        bo = bo[bo["stock_id"] == args.stock]
    elif len(bo) > args.n:
        bo = bo.sample(args.n, random_state=args.seed)
    bo = bo.sort_values("date")

    parts = []
    for k, row in enumerate(bo.itertuples(index=False)):
        fig = plot_case(panel, row, cfg.vcp)
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if k == 0 else False))
    html = ("<html><head><meta charset='utf-8'><title>VCP 人工檢查</title></head><body style='font-family:sans-serif'>"
            f"<h2>VCP 人工檢查（{len(bo)} 例，參數 min_swing_pct={cfg.vcp.min_swing_pct}）</h2>"
            "<p>請逐張判斷：這是不是你認知的 VCP？收縮是否逐段變小、量是否縮、pivot / 停損位置是否合理？"
            "灰色 × 是轉折的確認日，回測只用到這天以後。</p>" + "<hr>".join(parts) + "</body></html>")
    out = OUTPUT_DIR / ("vcp_review.html" if not args.stock else f"vcp_review_{args.stock}.html")
    out.write_text(html, encoding="utf-8")
    print(f"{len(bo)} 例 → {out}")


if __name__ == "__main__":
    main()
