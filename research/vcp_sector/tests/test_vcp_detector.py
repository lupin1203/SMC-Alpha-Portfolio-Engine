import numpy as np
import pandas as pd

from research.vcp_sector.config import VCPConfig
from research.vcp_sector.swing_points import zigzag
from research.vcp_sector.vcp_detector import detect_stock, prepare_arrays

CFG = VCPConfig()


def _path(points):
    """[(長度, 終點價, 成交量), ...] → 線性內插的收盤價與量。"""
    close, vol = [points[0][1]], [points[0][2]]
    for n, end, v in points[1:]:
        start = close[-1]
        close += list(np.linspace(start, end, n + 1)[1:])
        vol += [v] * n
    return np.array(close), np.array(vol, float)


def _build(depths=(0.25, 0.12, 0.06), last_vol=0.5e6, breakout_vol=3e6):
    pts = [(0, 50, 1e6), (260, 100, 1e6)]                     # 前置漲勢
    h = 100.0
    highs = [100.0, 98.0, 96.0]
    for k, d in enumerate(depths):
        v = last_vol if k == len(depths) - 1 else (1e6 if k == 0 else 0.8e6)
        pts.append((15 - 3 * k, highs[k] * (1 - d), v))           # 回檔
        nxt = highs[k + 1] if k + 1 < len(depths) else 95.5         # 反彈（最後一段停在 pivot 下）
        pts.append((12 - 2 * k, nxt, v))
    pts.append((1, 97.5, breakout_vol))                            # 突破日
    bo = sum(p[0] for p in pts)
    pts.append((10, 99, 1e6))
    close, vol = _path(pts)
    assert close[bo] == 97.5
    idx = pd.bdate_range("2023-01-02", periods=len(close))
    s = lambda x: pd.Series(x, index=idx)
    return idx, s(close * 1.005), s(close * 0.995), s(close), s(vol), bo


def _detect(idx, hi, lo, c, v, cut=None):
    if cut is not None:
        idx, hi, lo, c, v = idx[:cut + 1], hi.iloc[:cut + 1], lo.iloc[:cut + 1], c.iloc[:cut + 1], v.iloc[:cut + 1]
    a = prepare_arrays(hi, lo, c, v, CFG)
    sw = zigzag(a.high, a.low, CFG.min_swing_pct)
    return pd.DataFrame(detect_stock(idx, "X", a, sw, CFG))


def test_standard_vcp_detected():
    idx, hi, lo, c, v, bo = _build()
    res = _detect(idx, hi, lo, c, v)
    assert len(res) == 1
    r = res.iloc[0]
    assert r.date == idx[bo]
    assert r.n_contractions == 3
    assert abs(r.pivot - 96 * 1.005) < 1e-9                      # pivot = 最後一段收縮的高點（high）
    assert abs(r.stop_price - 96 * 0.94 * 0.995) < 1e-9           # stop = 最後一段低點（low）
    d = [float(x.strip("%")) / 100 for x in r.contraction_depths.split("/")]
    assert d[0] > d[1] > d[2] and d[0] < 0.35 and d[2] < 0.10


def test_expanding_pullbacks_not_detected():
    idx, hi, lo, c, v, _ = _build(depths=(0.06, 0.12, 0.25))
    assert _detect(idx, hi, lo, c, v).empty


def test_no_volume_dry_up_not_detected():
    idx, hi, lo, c, v, _ = _build(last_vol=1.2e6, breakout_vol=5e6)
    assert _detect(idx, hi, lo, c, v).empty


def test_weak_breakout_volume_not_detected():
    idx, hi, lo, c, v, _ = _build(breakout_vol=0.8e6)
    assert _detect(idx, hi, lo, c, v).empty


def test_no_lookahead_breakout_day_truncation():
    """資料截到突破日當天，仍要偵測到同一個突破（不需要未來資料）。"""
    idx, hi, lo, c, v, t = _build()
    full = _detect(idx, hi, lo, c, v)
    part = _detect(idx, hi, lo, c, v, cut=t)
    pd.testing.assert_frame_equal(full, part)
    assert _detect(idx, hi, lo, c, v, cut=t - 1).empty            # 前一天還沒有訊號


def test_non_shrinking_last_contraction_rejected():
    """最後一段 9% 雖然 <= 10%，但比前一段 8% 深 → 不是收縮。"""
    idx, hi, lo, c, v, _ = _build(depths=(0.20, 0.08, 0.09))
    assert _detect(idx, hi, lo, c, v).empty
    idx, hi, lo, c, v, _ = _build(depths=(0.20, 0.10, 0.05))   # 對照組：遞減就要抓到
    assert len(_detect(idx, hi, lo, c, v)) == 1
