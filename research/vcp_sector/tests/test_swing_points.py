import numpy as np
import pandas as pd

from research.vcp_sector.swing_points import find_swings, swings_known_at


def _bars(close, spread=0.0):
    c = np.asarray(close, float)
    idx = pd.bdate_range("2024-01-01", periods=len(c))
    return pd.DataFrame({"high": c * (1 + spread), "low": c * (1 - spread), "close": c}, index=idx)


def test_sawtooth_points_and_confirm_dates():
    # 100 → 110（高）→ 99（低）→ 108（高）→ 100
    close = [100, 104, 108, 110, 107, 104, 101, 99, 102, 105, 108, 104, 100]
    df = _bars(close)
    sw = find_swings(df, 0.05)
    d = df.index
    # 第一個轉折：起漲點 100 在 104×... 不到 5%？100→108 達 5% 於 idx2 → 確認低點 idx0
    assert list(sw["kind"]) == ["L", "H", "L", "H"]
    assert list(sw["price"]) == [100, 110, 99, 108]
    assert list(sw["date"]) == [d[0], d[3], d[7], d[10]]
    # 確認日：110×0.95=104.5 → idx5(104) 才確認；99×1.05=103.95 → idx9(105)；108×0.95=102.6 → idx12(100)
    assert list(sw["confirm_date"]) == [d[2], d[5], d[9], d[12]]
    assert (sw["confirm_date"] > sw["date"]).all()


def test_unconfirmed_last_extreme_not_reported():
    close = [100, 110, 120, 118]          # 120 只回 1.7%，未確認
    sw = find_swings(_bars(close), 0.05)
    assert "H" not in set(sw["kind"])


def test_no_lookahead_truncation_invariance():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 600)))
    df = _bars(close, spread=0.01)
    full = find_swings(df, 0.04)
    for cut in (150, 333, 599):
        t = df.index[cut]
        part = find_swings(df.iloc[: cut + 1], 0.04)
        pd.testing.assert_frame_equal(swings_known_at(full, t).reset_index(drop=True),
                                      part.reset_index(drop=True))


def test_alternating_and_nan_skipped():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    df = _bars(close, spread=0.01)
    df.iloc[50:55] = np.nan
    sw = find_swings(df, 0.04)
    k = sw["kind"].tolist()
    assert all(a != b for a, b in zip(k, k[1:]))       # H/L 交替
    assert not sw["idx"].between(50, 54).any()
