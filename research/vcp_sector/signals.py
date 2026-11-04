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


def signal_vcp(vcp: pd.DataFrame, strength: pd.DataFrame, groups: dict[str, list[str]],
               max_chase_pct: float = 0.05) -> pd.DataFrame:
    """訊號 A / C。

    - A_vcp：所有 VCP 突破
    - C_vcp_strong：突破日（t）所屬族群 is_strong（任一族群即可）
    - A_vcp_not_strong：突破日所屬族群「不是」is_strong（C 的對照組；A = C ∪ 這組）
    停損 = VCP 最後一段收縮低點；max_entry = pivot × (1 + max_chase_pct)，隔日開盤高於此價不追。
    """
    bo = vcp[vcp["is_breakout"]].copy()
    if bo.empty:
        return pd.DataFrame(columns=SIGNAL_COLS)
    strong = set(map(tuple, strength.loc[strength["is_strong"], ["date", "group"]].to_numpy()))
    s2g: dict[str, list[str]] = {}
    for g, ids in groups.items():
        for s in ids:
            s2g.setdefault(s, []).append(g)

    rows = []
    for r in bo.itertuples(index=False):
        gs = s2g.get(r.stock_id, [])
        strong_gs = [g for g in gs if (r.date, g) in strong]
        base = {"signal_date": r.date, "stock_id": r.stock_id, "stop_price": r.stop_price,
                "max_entry": r.pivot * (1 + max_chase_pct)}
        rows.append(base | {"group": gs[0] if gs else "", "variant": "A_vcp"})
        if strong_gs:
            rows.append(base | {"group": strong_gs[0], "variant": "C_vcp_strong"})
        else:
            rows.append(base | {"group": gs[0] if gs else "", "variant": "A_vcp_not_strong"})
    return pd.DataFrame(rows, columns=SIGNAL_COLS + ["max_entry"])
