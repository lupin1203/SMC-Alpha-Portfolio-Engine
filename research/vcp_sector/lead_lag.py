"""Phase 4（簡化版）：族群內的龍頭 / 落後候選。

- 龍頭：族群內近 20 日報酬最高、且 20 日均成交值在族群內排前 50% 的前 n_leaders 檔
- 落後候選：20 日報酬 < 族群中位數，且收盤仍在 MA50 之上（避免撿到弱勢股），
  取最接近中位數的 n_laggards 檔

只用 t 日（含）以前資料。lead-lag cross-correlation 依計畫排在最後，尚未實作。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, LeadLagConfig


class StockPanel:
    """寬表集合 + 個股衍生指標，回測與選股共用，避免重算。"""

    def __init__(self, prices: pd.DataFrame, eligible: pd.DataFrame,
                 cfg: LeadLagConfig = DEFAULT_CONFIG.leadlag, turnover_window: int = 20):
        wide = {f: prices.pivot(index="date", columns="stock_id", values=f).sort_index()
                for f in ("open", "high", "low", "close", "turnover")}
        self.dates = wide["close"].index
        self.stocks = wide["close"].columns
        for k, v in wide.items():
            setattr(self, k, v)
        self.eligible = eligible.reindex(index=self.dates, columns=self.stocks).fillna(False).astype(bool)
        self.ret_n = self.close.pct_change(cfg.ret_window, fill_method=None)
        self.ma_long = self.close.rolling(cfg.ma_long, min_periods=cfg.ma_long).mean()
        self.avg_turnover = self.turnover.rolling(turnover_window, min_periods=turnover_window).mean()
        self.col = {s: i for i, s in enumerate(self.stocks)}
        self.row = {d: i for i, d in enumerate(self.dates)}


def pick_leaders_laggards(panel: StockPanel, date, members: list[str],
                          cfg: LeadLagConfig = DEFAULT_CONFIG.leadlag) -> tuple[list[str], list[str]]:
    cols = [m for m in members if m in panel.col]
    if not cols:
        return [], []
    elig = panel.eligible.loc[date, cols]
    cols = elig.index[elig.to_numpy()].tolist()
    if not cols:
        return [], []
    ret = panel.ret_n.loc[date, cols].dropna()
    if ret.empty:
        return [], []

    turn = panel.avg_turnover.loc[date, ret.index]
    liquid = turn[turn.rank(pct=True) >= (1 - cfg.leader_turnover_pct)].index
    leaders = ret.loc[liquid].sort_values(ascending=False).head(cfg.n_leaders).index.tolist()

    med = ret.median()
    above_ma = panel.close.loc[date, ret.index] > panel.ma_long.loc[date, ret.index]
    lag = ret[(ret < med) & above_ma.to_numpy()]
    laggards = lag.sort_values(ascending=False).head(cfg.n_laggards).index.tolist()
    laggards = [s for s in laggards if s not in leaders]
    return leaders, laggards


def daily_leaders_laggards(panel: StockPanel, strength: pd.DataFrame, groups: dict[str, list[str]],
                           cfg: LeadLagConfig = DEFAULT_CONFIG.leadlag) -> pd.DataFrame:
    """每日、每個 is_strong 族群的龍頭 / 落後清單（給報表用）。"""
    rows = []
    for r in strength[strength["is_strong"]].itertuples():
        lead, lag = pick_leaders_laggards(panel, r.date, groups.get(r.group, []), cfg)
        rows.append({"date": r.date, "group": r.group, "leaders": lead, "laggards": lag})
    return pd.DataFrame(rows, columns=["date", "group", "leaders", "laggards"])


__all__ = ["StockPanel", "pick_leaders_laggards", "daily_leaders_laggards", "np"]
