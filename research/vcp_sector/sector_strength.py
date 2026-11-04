"""Phase 3：族群強度。

每個族群、每個交易日計算：

==================  ==========================================================
pct_above_ma20      當日有效成員中 close > MA20 的比例
pct_above_ma50      當日有效成員中 close > MA50 的比例
pct_new_high_20     當日有效成員中 close 創 20 日收盤新高（含當日）的比例
group_ret_20        族群等權指數 20 日報酬（每日以有效成員報酬等權平均後連乘）
group_rs            group_ret_20 − 大盤 20 日報酬
group_rank          group_rs 當日在所有（有效）族群中的百分位排名，最高 = 1.0
n_members           當日有效成員數
is_strong           pct_above_ma20 >= 0.6 且 group_rank >= 0.7
                    且 pct_above_ma20 > 5 個交易日前的值
==================  ==========================================================

所有指標在 t 日只使用 t 日（含）以前的收盤資料；可在 t 日收盤後算出、
最早於 t+1 日進場使用。當日有效成員 < min_members 的族群，該日所有指標為 NaN、
is_strong = False。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, SectorConfig

METRIC_COLS = ["pct_above_ma20", "pct_above_ma50", "pct_new_high_20",
               "group_ret_20", "group_rs", "group_rank", "n_members", "is_strong"]


def _stock_level_features(close: pd.DataFrame, cfg: SectorConfig) -> dict[str, pd.DataFrame]:
    ma_s = close.rolling(cfg.ma_short, min_periods=cfg.ma_short).mean()
    ma_l = close.rolling(cfg.ma_long, min_periods=cfg.ma_long).mean()
    hh = close.rolling(cfg.new_high_window, min_periods=cfg.new_high_window).max()
    return {
        # 比較時 NaN 會得到 False；分母用有效成員數，所以 MA 不足的成員視為「不在均線上」
        "above_ma_s": (close > ma_s),
        "above_ma_l": (close > ma_l),
        "new_high": (close >= hh) & hh.notna(),
        "ret_1d": close.pct_change(fill_method=None),
    }


def compute_sector_strength(
    close: pd.DataFrame,
    groups: dict[str, list[str]],
    benchmark: pd.Series,
    eligible: pd.DataFrame | None = None,
    cfg: SectorConfig = DEFAULT_CONFIG.sector,
) -> pd.DataFrame:
    """計算族群強度。

    Parameters
    ----------
    close : 寬表收盤價（index=date, columns=stock_id），應為還原權息價
    groups : ``{族群: [stock_id, ...]}``
    benchmark : 大盤收盤價 Series（index=date）
    eligible : 與 close 同形狀的布林寬表（逐日可交易池）；None 代表收盤價非 NaN 即有效

    Returns
    -------
    長表 ``DataFrame[date, group, pct_above_ma20, ..., is_strong]``
    """
    close = close.sort_index()
    if eligible is None:
        eligible = close.notna()
    eligible = eligible.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    feats = _stock_level_features(close, cfg)

    bench = benchmark.sort_index().reindex(close.index).ffill()
    bench_ret = bench.pct_change(cfg.ret_window, fill_method=None)

    per_group = {}
    for name, members in groups.items():
        cols = [c for c in members if c in close.columns]
        if not cols:
            continue
        elig = eligible[cols]
        n = elig.sum(axis=1)

        def frac(flag: pd.DataFrame) -> pd.Series:
            return (flag[cols] & elig).sum(axis=1) / n.replace(0, np.nan)

        # 等權指數：每日以「當日有效成員」的報酬平均，再滾動連乘
        r = feats["ret_1d"][cols].where(elig)
        idx_ret = r.mean(axis=1, skipna=True).fillna(0.0)
        ret_n = np.exp(np.log1p(idx_ret).rolling(cfg.ret_window, min_periods=cfg.ret_window).sum()) - 1

        g = pd.DataFrame({
            "pct_above_ma20": frac(feats["above_ma_s"]),
            "pct_above_ma50": frac(feats["above_ma_l"]),
            "pct_new_high_20": frac(feats["new_high"]),
            "group_ret_20": ret_n,
            "n_members": n,
        })
        # 有效成員數必須在整個報酬窗內都達標，否則報酬由太少檔股票決定
        enough = n.rolling(cfg.ret_window, min_periods=cfg.ret_window).min() >= cfg.min_members
        g.loc[~enough | (n < cfg.min_members),
              ["pct_above_ma20", "pct_above_ma50", "pct_new_high_20", "group_ret_20"]] = np.nan
        g["group_rs"] = g["group_ret_20"] - bench_ret
        per_group[name] = g

    if not per_group:
        return pd.DataFrame(columns=["date", "group"] + METRIC_COLS)

    out = pd.concat(per_group, names=["group", "date"])
    rs_wide = out["group_rs"].unstack("group")
    rank_wide = rs_wide.rank(axis=1, pct=True)  # NaN 族群不參與排名
    out["group_rank"] = rank_wide.stack().reindex(out.swaplevel().index).to_numpy()

    out["is_strong"] = False
    out = out.reset_index()[["date", "group"] + METRIC_COLS]
    out["n_members"] = out["n_members"].astype(int)
    out = out.sort_values(["date", "group"]).reset_index(drop=True)
    return apply_strong_rule(out, cfg)


def apply_strong_rule(strength: pd.DataFrame, cfg: SectorConfig = DEFAULT_CONFIG.sector) -> pd.DataFrame:
    """依 cfg 的門檻重新計算 is_strong（敏感度分析時不必重算其他指標）。"""
    out = strength.copy()
    breadth = out.pivot(index="date", columns="group", values="pct_above_ma20")
    rising = (breadth > breadth.shift(cfg.strong_breadth_lookback)).stack()
    rising = rising.reindex(pd.MultiIndex.from_frame(out[["date", "group"]])).fillna(False)
    out["is_strong"] = (
        (out["pct_above_ma20"] >= cfg.strong_pct_above_ma20)
        & (out["group_rank"] >= cfg.strong_rank)
        & rising.to_numpy().astype(bool)
    )
    return out


def strong_turn_events(strength: pd.DataFrame) -> pd.DataFrame:
    """族群「剛轉強」事件：is_strong 由 False 變 True 的那一天（訊號 B 會用到）。"""
    s = strength.sort_values(["group", "date"]).copy()
    prev = s.groupby("group")["is_strong"].shift(1, fill_value=False)
    return s[s["is_strong"] & ~prev.astype(bool)].reset_index(drop=True)
