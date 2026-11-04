"""Phase 2：族群分群。輸出 ``dict[group_name, list[stock_id]]``。

1. 官方產業別（baseline）
2. 自訂題材清單（groups.yaml，手動維護）
3. 相關係數分群 —— 依計畫最後再做，尚未實作

注意：官方產業別與 yaml 都是「現在」的分類，套用到過去會有輕微的倖存者偏差
（已下市股票不在清單內）。族群強度的成員資格另外經過逐日 eligibility mask。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .config import DEFAULT_CONFIG


def official_groups(stock_info: pd.DataFrame) -> dict[str, list[str]]:
    """官方產業別：``stock_info`` 需有 stock_id、industry_category。"""
    df = stock_info.dropna(subset=["industry_category"])
    return {
        str(cat): sorted(g["stock_id"].astype(str).unique().tolist())
        for cat, g in df.groupby("industry_category")
    }


def theme_groups(path: Path | str = DEFAULT_CONFIG.sector.groups_yaml,
                 universe: set[str] | None = None) -> dict[str, list[str]]:
    """讀 groups.yaml：``{題材: [stock_id, ...]}``。可用 ``universe`` 過濾掉不在資料中的代號。"""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    groups = {}
    for name, ids in raw.items():
        ids = [str(i).split(".")[0] for i in (ids or [])]
        if universe is not None:
            ids = [i for i in ids if i in universe]
        groups[str(name)] = sorted(set(ids))
    return groups


def correlation_groups(*_args, **_kwargs):
    """相關係數分群（120 日報酬、1−corr 距離、hierarchical clustering、每月重算）。

    依開發順序排在最後，Phase 1–3 尚未實作。
    """
    raise NotImplementedError("相關係數分群排在最後階段實作")


def stock_to_groups(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """反查：個股 → 所屬族群（一檔可屬於多個題材）。"""
    out: dict[str, list[str]] = {}
    for g, ids in groups.items():
        for s in ids:
            out.setdefault(s, []).append(g)
    return out
