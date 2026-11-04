import numpy as np
import pandas as pd
import pytest

from research.vcp_sector.config import DataConfig, SectorConfig
from research.vcp_sector.data_loader import eligibility_mask, filter_security_type
from research.vcp_sector.sector_groups import official_groups, theme_groups
from research.vcp_sector.sector_strength import compute_sector_strength, strong_turn_events
from research.vcp_sector.synthetic import make_synthetic_market

# 小視窗版設定，方便手算
SMALL = SectorConfig(min_members=2, ma_short=3, ma_long=4, new_high_window=3, ret_window=2,
                     strong_pct_above_ma20=0.6, strong_rank=0.7, strong_breadth_lookback=1)


def _toy():
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = pd.DataFrame({
        "A": [10, 11, 12, 13, 14, 15],   # 一路漲
        "B": [10, 10, 10, 10, 9, 8],     # 走弱
        "C": [10, 9, 8, 7, 6, 5],        # 一路跌
        "D": [10, 10, 10, 11, 12, 13],
    }, index=dates, dtype=float)
    bench = pd.Series([100, 100, 100, 100, 100, 100], index=dates, dtype=float)
    groups = {"G1": ["A", "B"], "G2": ["C", "D"]}
    return close, bench, groups


def test_hand_computed_breadth_and_returns():
    close, bench, groups = _toy()
    s = compute_sector_strength(close, groups, bench, cfg=SMALL).set_index(["date", "group"])
    d = close.index[-1]

    # MA3 at last day: A=14, B=9, C=6, D=12 → A above, B below (8<9), C below, D above
    assert s.loc[(d, "G1"), "pct_above_ma20"] == pytest.approx(0.5)
    assert s.loc[(d, "G2"), "pct_above_ma20"] == pytest.approx(0.5)
    # MA4: A=13.5 → above；B=9.25 → below
    assert s.loc[(d, "G1"), "pct_above_ma50"] == pytest.approx(0.5)
    # 3 日新高：A 15 是；B 否；C 否；D 13 是
    assert s.loc[(d, "G1"), "pct_new_high_20"] == pytest.approx(0.5)

    # 等權指數 2 日報酬：G1 日報酬 = mean(14/13-1, 9/10-1), mean(15/14-1, 8/9-1)
    r1 = np.mean([14 / 13 - 1, 9 / 10 - 1])
    r2 = np.mean([15 / 14 - 1, 8 / 9 - 1])
    assert s.loc[(d, "G1"), "group_ret_20"] == pytest.approx((1 + r1) * (1 + r2) - 1)
    # 大盤持平 → rs = ret
    assert s.loc[(d, "G1"), "group_rs"] == pytest.approx(s.loc[(d, "G1"), "group_ret_20"])

    # 兩個族群：rs 高者 rank = 1.0，低者 0.5
    hi = s.loc[d, "group_rs"].idxmax()
    assert s.loc[(d, hi), "group_rank"] == 1.0
    assert s.loc[d, "group_rank"].min() == 0.5


def test_min_members_excluded():
    close, bench, _ = _toy()
    s = compute_sector_strength(close, {"tiny": ["A"]}, bench, cfg=SMALL)
    assert s["pct_above_ma20"].isna().all()
    assert not s["is_strong"].any()


def test_is_strong_definition():
    close, bench, groups = _toy()
    s = compute_sector_strength(close, groups, bench, cfg=SMALL).sort_values(["group", "date"])
    prev = s.groupby("group")["pct_above_ma20"].shift(1)
    expected = (s["pct_above_ma20"] >= 0.6) & (s["group_rank"] >= 0.7) & (s["pct_above_ma20"] > prev)
    assert (s["is_strong"] == expected.fillna(False)).all()


def test_no_lookahead_truncation_invariance():
    """截斷未來資料後，過去每一天的指標必須完全一樣。"""
    prices, info, bench = make_synthetic_market(n_days=400, seed=1)
    groups = official_groups(info)
    close = prices.pivot(index="date", columns="stock_id", values="close")
    cfg = DataConfig(min_listed_days=60, min_avg_turnover=0)
    full = compute_sector_strength(close, groups, bench, eligibility_mask(prices, cfg))

    cut = close.index[250]
    p_cut = prices[prices["date"] <= cut]
    part = compute_sector_strength(close.loc[:cut], groups, bench.loc[:cut],
                                   eligibility_mask(p_cut, cfg))
    a = full[full["date"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, part.reset_index(drop=True))


def test_eligibility_point_in_time():
    prices, _, _ = make_synthetic_market(n_days=400, seed=2)
    cfg = DataConfig(min_listed_days=250, min_avg_turnover=0)
    m = eligibility_mask(prices, cfg)
    # 1101 是第 300 天才上市的那檔 → 不可能在 400 天內滿 250 日
    assert not m["1101"].any()
    # 1102 從第 0 天就有資料 → 第 250 個交易日（index 249）才開始合格
    assert not m["1102"].iloc[248] and m["1102"].iloc[249]


def test_turnover_filter():
    prices, _, _ = make_synthetic_market(n_days=300, seed=3)
    cfg = DataConfig(min_listed_days=1, min_avg_turnover=1e15)
    assert not eligibility_mask(prices, cfg).any().any()


def test_filter_security_type():
    info = pd.DataFrame({"stock_id": ["2330", "0050", "00878", "2330A", "030001", "1101"],
                         "industry_category": ["半導體業", "ETF", "ETF", "x", "x", "ETF"]})
    assert filter_security_type(info)["stock_id"].tolist() == ["2330"]


def test_theme_groups_yaml(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("AI: [2317, '2382.TW', 9999]\n空的:\n", encoding="utf-8")
    g = theme_groups(p, universe={"2317", "2382"})
    assert g == {"AI": ["2317", "2382"], "空的": []}


def test_strong_turn_events():
    df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5).repeat(1),
                       "group": "G", "is_strong": [False, True, True, False, True]})
    ev = strong_turn_events(df)
    assert ev["date"].tolist() == [df["date"][1], df["date"][4]]
