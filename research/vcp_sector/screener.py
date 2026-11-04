"""每日 VCP 候選清單（半自動：程式篩、你判斷）。

用法（收盤後，台北 14:30 以後）::

    python -m research.vcp_sector.screener                    # 今天
    python -m research.vcp_sector.screener --date 2024-06-19  # 重現過去某天（只用當天以前的資料）

輸出到 ``research/vcp_sector/output/journal/``：
- ``candidates_YYYYMMDD.csv``：候選清單，最後兩欄 ``decision``、``note`` 留給你填
  （decision 填「買」或「不買」；note 寫理由，例如「量縮不夠」「營收衰退」）
- ``candidates_YYYYMMDD.html``：每檔一張 K 線圖（swing 點、pivot、停損、追價上限）

兩種候選：
- ``breakout``：今天突破 pivot 且量增 → 明天開盤可進場（開盤 > 追價上限就不追）
- ``setup``：結構成立、量已縮、收盤還在 pivot 之下 → 觀察名單，等突破

已經填過 decision 的 CSV 不會被覆寫。累積一段時間後用 ``python -m research.vcp_sector.journal``
比較「你挑的」vs「你沒挑的」vs「全部」的實際結果。
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from .config import DEFAULT_CONFIG
from .lead_lag import StockPanel
from .vcp_detector import detect_all

JOURNAL_DIR_NAME = "journal"


def build_candidates(inp: dict, cfg=DEFAULT_CONFIG) -> tuple[pd.DataFrame, StockPanel]:
    panel = StockPanel(inp["prices"], inp["eligible"], cfg.leadlag, cfg.data.turnover_window)
    last = panel.dates[-1]
    vcp = detect_all(panel, cfg.vcp, setup_on_last_day=True)
    today = vcp[vcp["date"] == last].copy()
    if today.empty:
        return today, panel

    st = inp["strength"]
    st_last = st[st["date"] == last].set_index("group")
    s2g: dict[str, list[str]] = {}
    for g, ids in inp["groups"].items():
        for s in ids:
            s2g.setdefault(s, []).append(g)
    name = inp["info"].set_index("stock_id")["stock_name"] if "stock_name" in inp["info"] else pd.Series(dtype=str)

    rows = []
    for r in today.itertuples(index=False):
        g = (s2g.get(r.stock_id) or [""])[0]
        gi = st_last.loc[g] if g in st_last.index else None
        close = panel.close.at[last, r.stock_id]
        rows.append({
            "date": last.date(), "type": "breakout" if r.is_breakout else "setup",
            "stock_id": r.stock_id, "name": name.get(r.stock_id, ""),
            "close": round(close, 2), "pivot": round(r.pivot, 2),
            "max_entry": round(r.pivot * (1 + cfg.vcp.max_chase_pct), 2),
            "stop_price": round(r.stop_price, 2),
            "dist_to_pivot": round(close / r.pivot - 1, 4),
            "over_max_entry": bool(close > r.pivot * (1 + cfg.vcp.max_chase_pct)),  # 收盤已超過追價上限
            "risk_from_pivot": round(1 - r.stop_price / r.pivot, 4),
            "contractions": r.contraction_depths,
            "base_days": int(panel.row[last] - panel.row[r.base_start]),
            "ret_20d": round(panel.ret_n.at[last, r.stock_id], 4),
            "group": g,
            "group_is_strong": bool(gi["is_strong"]) if gi is not None else False,
            "group_rank": round(gi["group_rank"], 2) if gi is not None and pd.notna(gi["group_rank"]) else None,
            "decision": "", "note": "",
            # 給 journal / 圖表用（不需手動編輯）
            "base_start": r.base_start.date(), "is_breakout": r.is_breakout,
            "n_contractions": r.n_contractions,
        })
    df = pd.DataFrame(rows).sort_values(["type", "dist_to_pivot"], ascending=[True, False])
    return df, panel


def write_outputs(df: pd.DataFrame, panel: StockPanel, out_dir, cfg=DEFAULT_CONFIG) -> tuple:
    from .vcp_review import plot_case

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp(df["date"].iloc[0]).strftime("%Y%m%d")
    csv = out_dir / f"candidates_{tag}.csv"
    if csv.exists():
        old = pd.read_csv(csv, dtype={"stock_id": str}, encoding="utf-8-sig")
        if old["decision"].fillna("").astype(str).str.strip().ne("").any():
            print(f"⚠ {csv.name} 已有你填的 decision，不覆寫。")
        else:
            df.to_csv(csv, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(csv, index=False, encoding="utf-8-sig")

    parts = []
    for k, r in enumerate(df.itertuples(index=False)):
        row = pd.Series({"stock_id": r.stock_id, "date": pd.Timestamp(r.date), "base_start": pd.Timestamp(r.base_start),
                         "pivot": r.pivot, "stop_price": r.stop_price, "n_contractions": r.n_contractions,
                         "contraction_depths": r.contractions, "is_breakout": r.is_breakout})
        fig = plot_case(panel, next(pd.DataFrame([row]).itertuples(index=False)), cfg.vcp, pad_after=0)
        info = (f"<p><b>{r.stock_id} {r.name}</b>｜{'今日突破' if r.type == 'breakout' else '整理中（觀察）'}｜"
                f"收盤 {r.close}｜pivot {r.pivot}｜追價上限 {r.max_entry}｜停損 {r.stop_price}"
                f"（距 pivot {r.risk_from_pivot:.1%}）｜整理 {r.base_days} 日｜族群 {r.group}"
                f"{'（轉強）' if r.group_is_strong else ''}</p>")
        parts.append(info + fig.to_html(full_html=False, include_plotlyjs="cdn" if k == 0 else False))
    html = out_dir / f"candidates_{tag}.html"
    html.write_text("<html><head><meta charset='utf-8'><title>VCP 候選</title></head>"
                    "<body style='font-family:sans-serif'>"
                    f"<h2>VCP 候選 {df['date'].iloc[0]}（突破 {int((df['type'] == 'breakout').sum())}、"
                    f"整理中 {int((df['type'] == 'setup').sum())}）</h2>"
                    "<p>看完在 CSV 的 decision 欄填「買」或「不買」、note 寫理由。"
                    "突破股：明天開盤若高於追價上限就不追。</p>"
                    + "<hr>".join(parts) + "</body></html>", encoding="utf-8")
    return csv, html


def main(argv=None):
    from .report import OUTPUT_DIR, load_inputs

    p = argparse.ArgumentParser(description="每日 VCP 候選清單")
    p.add_argument("--date", default=None, help="重現過去某天（YYYY-MM-DD）")
    p.add_argument("--no-api", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = DEFAULT_CONFIG

    inp = load_inputs("official", False, not args.no_api, cfg, as_of=args.date)
    df, panel = build_candidates(inp, cfg)
    last = panel.dates[-1]
    if df.empty:
        print(f"{last:%Y-%m-%d}：沒有符合條件的 VCP 突破或整理中個股。")
        return
    csv, html = write_outputs(df, panel, OUTPUT_DIR / JOURNAL_DIR_NAME, cfg)
    pd.set_option("display.width", 220)
    print(df[["type", "stock_id", "name", "close", "pivot", "max_entry", "over_max_entry", "stop_price",
              "dist_to_pivot",
              "contractions", "base_days", "group", "group_is_strong"]].to_string(index=False))
    print(f"\nCSV  → {csv}\nHTML → {html}")


if __name__ == "__main__":
    main()
