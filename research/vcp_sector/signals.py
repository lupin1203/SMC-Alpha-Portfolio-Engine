"""Phase 7：訊號。目前只有訊號 B（純族群強度）；A / C 等 VCP 完成後加入。

訊號列格式：``DataFrame[signal_date, stock_id, group, variant, stop_price]``
- signal_date 是 t 日（收盤後可知），進場一律 t+1 開盤，由 backtest 處理
- stop_price 只用 t 日（含）以前資料
"""
from __future__ import annotations

import pandas as pd

from .config import DEFAULT_CONFIG, LeadLagConfig, TradeConfig
from .lead_lag import StockPanel, pick_leaders_laggards
from .sector_strength import strong_turn_events

SIGNAL_COLS = ["signal_date", "stock_id", "group", "variant", "stop_price"]


def lowest_low_stop(panel: StockPanel, lookback: int) -> pd.DataFrame:
    return panel.low.rolling(lookback, min_periods=lookback).min()


def signal_b(panel: StockPanel, strength: pd.DataFrame, groups: dict[str, list[str]],
             lcfg: LeadLagConfig = DEFAULT_CONFIG.leadlag,
             tcfg: TradeConfig = DEFAULT_CONFIG.trade, cache: dict | None = None) -> pd.DataFrame:
    """訊號 B：族群「剛轉強」當日，買進族群內龍頭（B_leader）或落後候選（B_laggard）。

    cache：(date, group) → (leaders, laggards)，跑參數網格時共用（選股與族群門檻無關）。
    """
    cache = {} if cache is None else cache
    events = strong_turn_events(strength)
    stop_tbl = lowest_low_stop(panel, tcfg.stop_lookback)
    rows = []
    for ev in events.itertuples():
        if ev.date not in panel.row:
            continue
        key = (ev.date, ev.group)
        if key not in cache:
            cache[key] = pick_leaders_laggards(panel, ev.date, groups.get(ev.group, []), lcfg)
        leaders, laggards = cache[key]
        for variant, picks in (("B_leader", leaders), ("B_laggard", laggards)):
            for s in picks:
                rows.append({"signal_date": ev.date, "stock_id": s, "group": ev.group,
                             "variant": variant, "stop_price": stop_tbl.at[ev.date, s]})
    sig = pd.DataFrame(rows, columns=SIGNAL_COLS)
    # 同一天同一檔被多個族群選到 → 只留一筆
    return sig.drop_duplicates(["signal_date", "stock_id", "variant"]).reset_index(drop=True)
