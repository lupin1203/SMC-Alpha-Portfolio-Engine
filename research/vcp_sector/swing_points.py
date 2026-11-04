"""Phase 5：波段高低點（ZigZag 百分比法），**嚴格無前視**。

規則（逐根 K 棒、只用當根及以前的資料）：
- 目前在找高點時，持續記錄「進行中的最高價」(候選高點)。
  某根 K 棒的 low <= 候選高點 × (1 − min_swing_pct) → **確認**候選高點為 swing High，
  確認日 = 這根 K 棒的日期；之後轉為找低點，候選低點從這根的 low 開始。
- 找低點時對稱：high >= 候選低點 × (1 + min_swing_pct) → 確認 swing Low。
- 同一根 K 棒先檢查「是否反轉」（用前一刻的候選值），沒反轉才用這根更新候選值。
  日 K 看不到盤中先後順序，這樣處理比較保守（不會用同一根的高點又用它的低點確認）。

輸出每個轉折點：``date``（轉折實際發生日）、``price``、``kind``（'H'/'L'）、
``confirm_date``（確認日）。**回測與訊號只能使用 confirm_date <= t 的轉折點。**
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SWING_COLS = ["date", "price", "kind", "confirm_date", "idx", "confirm_idx"]


def zigzag(high: np.ndarray, low: np.ndarray, min_swing_pct: float) -> list[tuple]:
    """核心演算法（numpy 陣列版）。回傳 [(idx, price, kind, confirm_idx), ...]。

    NaN（停牌）的 K 棒直接略過。
    """
    n = len(high)
    out: list[tuple] = []
    up = 1 + min_swing_pct
    dn = 1 - min_swing_pct

    # 起始方向未定：同時追蹤最高與最低，先達到門檻的方向決定第一個轉折
    hi_i = lo_i = -1
    hi = -np.inf
    lo = np.inf
    mode = 0  # 0 = 未定, 1 = 找高點, -1 = 找低點
    for i in range(n):
        h, l_ = high[i], low[i]
        if not (np.isfinite(h) and np.isfinite(l_)):
            continue
        if mode == 0:
            if hi_i >= 0 and l_ <= hi * dn:          # 從最高點跌下來 → 先確認一個高點
                out.append((hi_i, hi, "H", i))
                mode, lo, lo_i = -1, l_, i
                continue
            if lo_i >= 0 and h >= lo * up:           # 從最低點漲上去 → 先確認一個低點
                out.append((lo_i, lo, "L", i))
                mode, hi, hi_i = 1, h, i
                continue
            if h > hi:
                hi, hi_i = h, i
            if l_ < lo:
                lo, lo_i = l_, i
        elif mode == 1:
            if l_ <= hi * dn:
                out.append((hi_i, hi, "H", i))
                mode, lo, lo_i = -1, l_, i
            elif h > hi:
                hi, hi_i = h, i
        else:
            if h >= lo * up:
                out.append((lo_i, lo, "L", i))
                mode, hi, hi_i = 1, h, i
            elif l_ < lo:
                lo, lo_i = l_, i
    return out


def find_swings(df: pd.DataFrame, min_swing_pct: float = 0.04) -> pd.DataFrame:
    """單檔股票的轉折點。``df`` 需有 high、low 欄，index 為日期。"""
    raw = zigzag(df["high"].to_numpy(float), df["low"].to_numpy(float), min_swing_pct)
    if not raw:
        return pd.DataFrame(columns=SWING_COLS)
    idx, price, kind, cidx = map(list, zip(*raw))
    dates = df.index
    return pd.DataFrame({"date": dates[idx], "price": price, "kind": kind,
                         "confirm_date": dates[cidx], "idx": idx, "confirm_idx": cidx})


def swings_known_at(swings: pd.DataFrame, t) -> pd.DataFrame:
    """t 日收盤時「已經知道」的轉折點（confirm_date <= t）。"""
    return swings[swings["confirm_date"] <= t]
