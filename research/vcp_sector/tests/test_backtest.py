import numpy as np
import pandas as pd
import pytest

from research.vcp_sector.backtest import simulate_trades, walk_forward, simulate_portfolio
from research.vcp_sector.config import DataConfig, TradeConfig, Config
from research.vcp_sector.data_loader import eligibility_mask
from research.vcp_sector.lead_lag import StockPanel
from research.vcp_sector.sector_groups import official_groups
from research.vcp_sector.sector_strength import compute_sector_strength
from research.vcp_sector.signals import signal_b
from research.vcp_sector.synthetic import make_synthetic_market

NOCOST = TradeConfig(fee_rate=0, tax_rate=0, slippage=0, exit_ma=3, trail_pct=0.10,
                     min_risk_pct=0.0, max_risk_pct=0.5, max_hold_days=100)


def _panel(o, h, l, c):
    dates = pd.bdate_range("2024-01-01", periods=len(c))
    df = pd.DataFrame({"date": dates, "stock_id": "X", "open": o, "high": h, "low": l, "close": c,
                       "volume": 1e6})
    df["turnover"] = df["close"] * df["volume"]
    elig = pd.DataFrame({"X": True}, index=dates)
    return StockPanel(df, elig), dates


def _sig(date, stop):
    return pd.DataFrame([{"signal_date": date, "stock_id": "X", "group": "G", "variant": "B", "stop_price": stop}])


def test_entry_next_open_and_stop_hit():
    p, d = _panel(o=[10, 10, 11, 10.5, 9.0], h=[10, 10, 11.5, 10.6, 9.2],
                  l=[10, 10, 10.8, 9.4, 8.8], c=[10, 10, 11.2, 9.6, 9.0])
    t = simulate_trades(p, _sig(d[1], 9.5), NOCOST).iloc[0]
    assert t.entry_date == d[2] and t.entry_price == 11          # t+1 開盤
    assert t.exit_reason == "stop" and t.exit_date == d[3] and t.exit_price == 9.5
    assert t.r == pytest.approx((9.5 - 11) / (11 - 9.5))          # = -1R


def test_gap_below_stop_exits_at_open():
    p, d = _panel(o=[10, 10, 11, 9.0], h=[10, 10, 11.5, 9.2], l=[10, 10, 10.8, 8.8], c=[10, 10, 11.2, 9.0])
    t = simulate_trades(p, _sig(d[1], 9.5), NOCOST).iloc[0]
    assert t.exit_price == 9.0 and t.r < -1                       # 跳空：虧損超過 1R


def test_trailing_exit_executes_next_open():
    c = [10, 10, 12, 14, 12.4, 12.0]
    o = [10, 10, 11, 13, 13.0, 12.2]
    p, d = _panel(o=o, h=[x + 0.2 for x in np.maximum(o, c)], l=[x - 0.2 for x in np.minimum(o, c)], c=c)
    t = simulate_trades(p, _sig(d[1], 7), NOCOST.__class__(**{**NOCOST.__dict__, "exit_ma": 50})).iloc[0]
    # 最高收盤 14，12.4 < 14×0.9=12.6 → d[4] 收盤觸發，d[5] 開盤 12.2 出場
    assert t.exit_reason == "trail" and t.exit_date == d[5] and t.exit_price == 12.2


def test_no_reentry_while_open():
    p, d = _panel(o=[10] * 8, h=[10.1] * 8, l=[9.9] * 8, c=[10] * 8)
    sig = pd.concat([_sig(d[1], 5), _sig(d[2], 5)])
    assert len(simulate_trades(p, sig, NOCOST)) == 1


def test_signals_no_lookahead():
    """截掉未來資料後，過去的訊號（含停損價）必須完全相同。"""
    prices, info, bench = make_synthetic_market(n_days=600, seed=5)
    groups = official_groups(info)
    dcfg = DataConfig(min_listed_days=60, min_avg_turnover=0)

    def run(px, b):
        elig = eligibility_mask(px, dcfg)
        close = px.pivot(index="date", columns="stock_id", values="close")
        st = compute_sector_strength(close, groups, b, elig)
        return signal_b(StockPanel(px, elig), st, groups)

    full = run(prices, bench)
    cut = sorted(prices["date"].unique())[450]
    part = run(prices[prices["date"] <= cut], bench.loc[:cut])
    a = full[full["signal_date"] <= cut].reset_index(drop=True)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, part.reset_index(drop=True))


def test_walk_forward_is_excludes_trades_exiting_in_oos():
    def tr(sig, exit_, r, key):
        return {"signal_date": pd.Timestamp(sig), "exit_date": pd.Timestamp(exit_), "r": r,
                "variant": "B", "key": key}
    rows = []
    # key A：IS 期間表現普通；但有一筆 2022-12 進場、2023 出場的大賺（不可被 IS 看見）
    for y in (2020, 2021, 2022):
        rows += [tr(f"{y}-03-01", f"{y}-04-01", 0.1, "A")] * 30
        rows += [tr(f"{y}-03-01", f"{y}-04-01", 0.2, "B")] * 30
    rows += [tr("2022-12-20", "2023-02-01", 100.0, "A")]
    rows += [tr("2023-03-01", "2023-04-01", 1.0, "A"), tr("2023-03-01", "2023-04-01", -1.0, "B")]
    df = pd.DataFrame(rows)
    grid = {k: g.drop(columns="key") for k, g in df.groupby("key")}
    wf, _ = walk_forward(grid, fixed_key="A", cfg=Config())
    row = wf[wf["oos_year"] == 2023].iloc[0]
    assert row["chosen"] == "B"                     # 若偷看到 100R 就會選 A
    assert row["oos_avg_r"] == -1.0 and row["fixed_oos_avg_r"] == 1.0


def test_portfolio_risk_sizing():
    p, d = _panel(o=[100] * 6, h=[101] * 6, l=[99] * 6, c=[100] * 6)
    trades = pd.DataFrame([{"signal_date": d[0], "stock_id": "X", "group": "G", "variant": "B",
                            "entry_date": d[1], "entry_price": 100.0, "stop_price": 90.0,
                            "exit_date": d[4], "exit_price": 100.0, "exit_reason": "x",
                            "hold_days": 3, "ret": 0.0, "r": 0.0}])
    cfg = TradeConfig(fee_rate=0, tax_rate=0, slippage=0, initial_capital=1_000_000,
                      risk_per_trade=0.01, max_position_pct=1.0)
    eq, taken = simulate_portfolio(p, trades, cfg)
    assert taken.iloc[0]["shares"] == 1000          # 1% × 100 萬 / 10 元風險 = 1000 股
    assert eq.iloc[-1] == pytest.approx(1_000_000)


def test_portfolio_same_day_stop_is_closed():
    """進場當日就停損的部位必須當日出場，不能殘留在投組裡。"""
    c = [100, 100, 95, 120, 150, 200]
    p, d = _panel(o=[100] * 6, h=[101] * 6, l=[94] * 6, c=c)
    trades = pd.DataFrame([{"signal_date": d[0], "stock_id": "X", "group": "G", "variant": "B",
                            "entry_date": d[2], "entry_price": 100.0, "stop_price": 95.0,
                            "exit_date": d[2], "exit_price": 95.0, "exit_reason": "stop",
                            "hold_days": 1, "ret": -0.05, "r": -1.0}])
    cfg = TradeConfig(fee_rate=0, tax_rate=0, slippage=0, initial_capital=1_000_000,
                      risk_per_trade=0.01, max_position_pct=1.0)
    eq, _ = simulate_portfolio(p, trades, cfg)
    assert eq.iloc[-1] == pytest.approx(990_000)    # 虧 1%，之後漲到 200 與它無關


def test_portfolio_equity_matches_trade_pnl():
    """不重疊的多筆交易：投組最終權益 = 逐筆損益加總（固定股數下）。"""
    prices, info, bench = make_synthetic_market(n_days=500, seed=9)
    elig = eligibility_mask(prices, DataConfig(min_listed_days=60, min_avg_turnover=0))
    panel = StockPanel(prices, elig)
    groups = official_groups(info)
    st = compute_sector_strength(panel.close, groups, bench, elig)
    tr = simulate_trades(panel, signal_b(panel, st, groups))
    tr = tr[tr["variant"] == "B_leader"].sort_values("entry_date")
    keep, last_exit = [], pd.Timestamp.min        # 取不重疊的交易
    for t in tr.itertuples():
        if t.entry_date > last_exit:
            keep.append(t.Index); last_exit = t.exit_date
    tr = tr.loc[keep]
    cfg = TradeConfig(max_positions=1, max_position_pct=1.0, risk_per_trade=0.001)
    eq, taken = simulate_portfolio(panel, tr, cfg)
    pnl = (taken["shares"] * (taken["exit_price"] * (1 - cfg.slippage) * (1 - cfg.fee_rate - cfg.tax_rate)
                              - taken["entry_price"] * (1 + cfg.fee_rate))).sum()
    assert eq.iloc[-1] == pytest.approx(cfg.initial_capital + pnl, rel=1e-9)


def test_no_chase_above_max_entry():
    p, d = _panel(o=[10, 10, 11, 11], h=[10.2, 10.2, 11.2, 11.2], l=[9.8, 9.8, 10.8, 10.8], c=[10, 10, 11, 11])
    sig = _sig(d[1], 9.0).assign(max_entry=10.5)          # 隔日開盤 11 > 10.5 → 不追
    assert simulate_trades(p, sig, NOCOST).empty
    sig = _sig(d[1], 9.0).assign(max_entry=11.5)
    assert len(simulate_trades(p, sig, NOCOST)) == 1


def test_fixed_close_exit_and_pct_stop():
    cfg = TradeConfig(fee_rate=0, tax_rate=0, slippage=0, exit_mode="fixed_close", fixed_hold_days=3)
    p, d = _panel(o=[10, 10, 10, 11, 12, 13, 14], h=[10.5] * 3 + [11.5, 12.5, 13.5, 14.5],
                  l=[9.5] * 3 + [10.5, 11.5, 12.5, 13.5], c=[10, 10, 10.2, 11.2, 12.2, 13.2, 14.2])
    sig = _sig(d[1], np.nan).assign(stop_pct=0.10)
    t = simulate_trades(p, sig, cfg).iloc[0]
    assert t.entry_date == d[2] and t.stop_price == pytest.approx(9.0)
    assert t.exit_reason == "time_close" and t.exit_date == d[4] and t.exit_price == 12.2   # 第 3 日收盤


def test_intraday_stop_does_not_free_slot_for_same_day_open():
    p, d = _panel(o=[100] * 6, h=[101] * 6, l=[99] * 6, c=[100] * 6)
    base = {"signal_date": d[0], "group": "G", "variant": "B", "entry_price": 100.0, "stop_price": 90.0,
            "exit_price": 90.0, "hold_days": 1, "ret": 0.0, "r": -1.0}
    tr = pd.DataFrame([
        base | {"stock_id": "X", "entry_date": d[1], "exit_date": d[3], "exit_reason": "stop"},
        base | {"stock_id": "X", "entry_date": d[3], "exit_date": d[5], "exit_reason": "time_close"},
    ])
    cfg = TradeConfig(fee_rate=0, tax_rate=0, slippage=0, max_positions=1, sizing="equal")
    _, taken = simulate_portfolio(p, tr, cfg)
    assert len(taken) == 1          # d[3] 開盤時第一筆還沒停損，名額未釋放
    tr.loc[0, "exit_reason"] = "trail"    # 若是「開盤出場」就可以在同日開盤換股
    _, taken = simulate_portfolio(p, tr, cfg)
    assert len(taken) == 2
