"""檢驗你的人工判斷有沒有加分。

讀 ``output/journal/candidates_*.csv`` 裡所有 ``type == breakout`` 的候選，
用跟回測完全相同的規則（隔日開盤進場、開盤 > 追價上限不追、停損 = 最後收縮低點、
移動停利 / 跌破 MA20 / 時間停損、含成本）模擬，依你填的 decision 分組比較。

用法：python -m research.vcp_sector.journal

- 「買」組平均 R 明顯高於「不買」組 → 你的判斷有價值
- 兩組差不多 → 判斷沒有額外資訊，照程式全買（或全不買）即可
- 樣本少於 30 筆時，任何差異都可能是運氣
- 還沒出場的交易以最新收盤計（exit_reason = open）
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from .backtest import diff_test, simulate_trades, summarize
from .config import DEFAULT_CONFIG
from .lead_lag import StockPanel

BUY, SKIP = "買", "不買"


def load_journal(journal_dir) -> pd.DataFrame:
    files = sorted(journal_dir.glob("candidates_*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype={"stock_id": str}, encoding="utf-8-sig") for f in files],
                   ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["decision"] = df["decision"].fillna("").astype(str).str.strip()
    df.loc[~df["decision"].isin([BUY, SKIP]), "decision"] = "未填"
    return df


def evaluate(journal: pd.DataFrame, panel: StockPanel, cfg=DEFAULT_CONFIG) -> pd.DataFrame:
    bo = journal[journal["type"] == "breakout"]
    sig = pd.DataFrame({"signal_date": bo["date"], "stock_id": bo["stock_id"], "group": bo["group"].fillna(""),
                        "variant": bo["decision"], "stop_price": bo["stop_price"], "max_entry": bo["max_entry"]})
    return simulate_trades(panel, sig, cfg.trade)


def main(argv=None):
    from .report import OUTPUT_DIR, load_inputs
    from .screener import JOURNAL_DIR_NAME

    p = argparse.ArgumentParser(description="人工判斷績效檢驗")
    p.add_argument("--no-api", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 200)
    cfg = DEFAULT_CONFIG

    journal = load_journal(OUTPUT_DIR / JOURNAL_DIR_NAME)
    if journal.empty:
        print("還沒有任何 candidates_*.csv，請先每天跑 screener 並填 decision。")
        return
    bo = journal[journal["type"] == "breakout"]
    print(f"紀錄：{journal['date'].min():%Y-%m-%d} ~ {journal['date'].max():%Y-%m-%d}，"
          f"突破候選 {len(bo)} 筆（買 {(bo['decision'] == BUY).sum()}、不買 {(bo['decision'] == SKIP).sum()}、"
          f"未填 {(bo['decision'] == '未填').sum()}）")

    inp = load_inputs("official", False, not args.no_api, cfg)
    panel = StockPanel(inp["prices"], inp["eligible"], cfg.leadlag, cfg.data.turnover_window)
    tr = evaluate(journal, panel, cfg)
    if tr.empty:
        print("還沒有可以評估的交易（訊號隔天才進場）。")
        return
    out = OUTPUT_DIR / JOURNAL_DIR_NAME / "journal_trades.csv"
    tr.to_csv(out, index=False, encoding="utf-8-sig")
    tbl = tr.groupby("variant").apply(summarize, include_groups=False)
    tbl.loc["全部候選"] = summarize(tr)
    print(tbl[["n", "win_rate", "avg_r", "median_r", "profit_factor", "t_stat"]]
          .to_string(float_format=lambda x: f"{x:,.3f}"))
    n_skipped_chase = len(bo) - len(tr)
    if n_skipped_chase:
        print(f"（另有 {n_skipped_chase} 筆因隔日開盤高於追價上限、停損距離不合理或尚未進場而未計入）")
    if {BUY, SKIP} <= set(tr["variant"]):
        d = diff_test(tr.loc[tr["variant"] == BUY, "r"], tr.loc[tr["variant"] == SKIP, "r"])
        print(f"\n買 − 不買：{d['diff']:+.3f}R（95% CI {d['ci_low']:+.3f} ~ {d['ci_high']:+.3f}；"
              f"n={d['n_a']} vs {d['n_b']}）")
        if min(d["n_a"], d["n_b"]) < 30:
            print("樣本 < 30，差異很可能是運氣，繼續累積。")
    print(f"\n明細 → {out}")


if __name__ == "__main__":
    main()
