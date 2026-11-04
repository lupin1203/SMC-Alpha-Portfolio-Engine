"""Phase 6：VCP 型態辨識。

在 t 日只使用：t 日（含）以前的 K 棒，以及 confirm_date <= t 的 swing 點。

1. 前置趨勢：close > MA50 > MA150 > MA200、MA200 高於 20 日前、
   close 距 52 週高點 <= 25%、距 52 週低點 >= 30%
2. 收縮結構：近 lookback 日內最高的 swing High 為整理起點 H0，之後每組 (swing High → swing Low)
   為一段收縮；取最後 2–4 段，深度逐段遞減（<= 前段 × contraction_ratio），
   第一段 <= 35%，最後一段 <= 10%；最後一個已確認轉折必須是 Low（收縮已完成）
3. 量縮：最後一段收縮（H→L）期間均量 < H0 之前 50 日均量 × vol_dry_ratio
4. pivot = 最後一段收縮的 swing High；stop = 最後一段收縮的 swing Low
5. 突破：close > pivot、前一日 close <= pivot（首次突破）、
   當日量 > 前 50 日（不含當日）均量 × breakout_vol_mult，且最後低點之後沒有跌破 stop
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, VCPConfig
from .swing_points import zigzag

VCP_COLS = ["date", "stock_id", "is_vcp_setup", "is_breakout", "pivot", "stop_price",
            "n_contractions", "contraction_depths", "base_start", "last_low_date"]


@dataclass
class StockArrays:
    """單檔股票的 numpy 陣列與預先算好的指標（全部是 t 日含以前的滾動值）。"""
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    trend_ok: np.ndarray
    avg_vol_prev: np.ndarray   # 前 N 日均量（不含當日）


def prepare_arrays(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                   cfg: VCPConfig = DEFAULT_CONFIG.vcp) -> StockArrays:
    ma_f = close.rolling(cfg.ma_fast, min_periods=cfg.ma_fast).mean()
    ma_m = close.rolling(cfg.ma_mid, min_periods=cfg.ma_mid).mean()
    ma_s = close.rolling(cfg.ma_slow, min_periods=cfg.ma_slow).mean()
    hi52 = high.rolling(252, min_periods=200).max()
    lo52 = low.rolling(252, min_periods=200).min()
    trend = ((close > ma_f) & (ma_f > ma_m) & (ma_m > ma_s)
             & (ma_s > ma_s.shift(cfg.ma_slow_rising_days))
             & (close >= hi52 * (1 - cfg.max_below_52w_high))
             & (close >= lo52 * (1 + cfg.min_above_52w_low)))
    avg_vol = volume.rolling(cfg.breakout_vol_days, min_periods=cfg.breakout_vol_days).mean().shift(1)
    return StockArrays(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float),
                       volume.to_numpy(float), trend.fillna(False).to_numpy(bool), avg_vol.to_numpy(float))


def analyze_base(a: StockArrays, swings: list[tuple], t: int, cfg: VCPConfig = DEFAULT_CONFIG.vcp):
    """t 日的收縮結構。swings = zigzag() 輸出（全期間），此處只取 confirm_idx <= t 者。

    回傳 dict（結構成立）或 None。
    """
    known = [s for s in swings if s[3] <= t and s[0] >= t - cfg.lookback]
    if len(known) < 2 or known[-1][2] != "L":
        return None
    highs = [k for k, s in enumerate(known) if s[2] == "H"]
    if not highs:
        return None
    h0 = max(highs, key=lambda k: known[k][1])
    seq = known[h0:]
    pairs = [(seq[k], seq[k + 1]) for k in range(len(seq) - 1) if seq[k][2] == "H" and seq[k + 1][2] == "L"]
    if len(pairs) < cfg.min_contractions:
        return None
    pairs = pairs[-cfg.max_contractions:]
    depths = [1 - lo[1] / hi[1] for hi, lo in pairs]
    if depths[0] > cfg.max_first_depth or depths[-1] > cfg.max_last_depth:
        return None
    if any(d > prev * cfg.contraction_ratio for prev, d in zip(depths, depths[1:])):
        return None

    base_i = pairs[0][0][0]
    last_h, last_l = pairs[-1]
    pre = a.volume[max(0, base_i - cfg.pre_base_vol_days):base_i]
    pre = pre[np.isfinite(pre)]
    last_seg = a.volume[last_h[0]:last_l[0] + 1]
    last_seg = last_seg[np.isfinite(last_seg)]
    if len(pre) < cfg.pre_base_vol_days * 0.8 or len(last_seg) == 0:
        return None
    vol_dry = last_seg.mean() < pre.mean() * cfg.vol_dry_ratio
    return {"pivot": last_h[1], "stop_price": last_l[1], "depths": depths, "base_i": base_i,
            "last_low_i": last_l[0], "vol_dry": bool(vol_dry)}


def _no_undercut(a: StockArrays, base: dict, t: int) -> bool:
    lows = a.low[base["last_low_i"] + 1:t + 1]
    lows = lows[np.isfinite(lows)]
    return len(lows) == 0 or lows.min() >= base["stop_price"]


def _row(dates, stock_id, t, base, setup, breakout):
    return {"date": dates[t], "stock_id": stock_id, "is_vcp_setup": setup, "is_breakout": breakout,
            "pivot": base["pivot"], "stop_price": base["stop_price"],
            "n_contractions": len(base["depths"]),
            "contraction_depths": "/".join(f"{d:.1%}" for d in base["depths"]),
            "base_start": dates[base["base_i"]], "last_low_date": dates[base["last_low_i"]]}


def detect_stock(dates: pd.DatetimeIndex, stock_id: str, a: StockArrays, swings: list[tuple],
                 cfg: VCPConfig = DEFAULT_CONFIG.vcp, eligible: np.ndarray | None = None,
                 setup_dates_idx: list[int] | None = None) -> list[dict]:
    """回傳突破日（is_breakout）＋指定日期（setup_dates_idx，例如最新一日）的 setup 狀態。"""
    rows = []
    c, v = a.close, a.volume
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    cand = (a.trend_ok & np.isfinite(c) & np.isfinite(a.avg_vol_prev)
            & (v > a.avg_vol_prev * cfg.breakout_vol_mult) & (c > prev_c))
    if eligible is not None:
        cand &= eligible
    for t in np.flatnonzero(cand):
        base = analyze_base(a, swings, t, cfg)
        if base is None or not base["vol_dry"]:
            continue
        if c[t] > base["pivot"] and prev_c[t] <= base["pivot"] and _no_undercut(a, base, t):
            rows.append(_row(dates, stock_id, t, base, True, True))
    for t in setup_dates_idx or []:
        if not a.trend_ok[t] or not np.isfinite(c[t]):
            continue
        base = analyze_base(a, swings, t, cfg)
        if base and base["vol_dry"] and c[t] <= base["pivot"] and _no_undercut(a, base, t):
            rows.append(_row(dates, stock_id, t, base, True, False))
    return rows


def detect_all(panel, cfg: VCPConfig = DEFAULT_CONFIG.vcp, swings_cache: dict | None = None,
               setup_on_last_day: bool = True) -> pd.DataFrame:
    """全市場 VCP 突破（只在可交易日）＋最新一日的 setup 清單。

    swings_cache：{(stock_id, min_swing_pct): swings}，跑參數網格時共用。
    """
    swings_cache = {} if swings_cache is None else swings_cache
    dates = panel.dates
    elig = panel.eligible.to_numpy()
    rows = []
    for j, s in enumerate(panel.stocks):
        a = prepare_arrays(panel.high[s], panel.low[s], panel.close[s], panel.volume[s], cfg)
        key = (s, cfg.min_swing_pct)
        if key not in swings_cache:
            swings_cache[key] = zigzag(a.high, a.low, cfg.min_swing_pct)
        last = [len(dates) - 1] if setup_on_last_day and elig[-1, j] else None
        rows += detect_stock(dates, s, a, swings_cache[key], cfg, elig[:, j], last)
    return pd.DataFrame(rows, columns=VCP_COLS)
