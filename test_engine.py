# test_engine.py
import unittest
import pandas as pd
from strategy import SMCStrategy

class TestSMCStrategy(unittest.TestCase):
    def setUp(self):
        # 策略至少需要 60 根 K 棒，前面先墊 60 根平盤，最後 4 根模擬完美缺口
        pad = 60
        data = {
            'Open':  [95] * pad + [100, 105, 102, 107],
            'High':  [96] * pad + [102, 106, 104, 115],
            'Low':   [94] * pad + [98,  103, 101, 108],
            'Close': [95] * pad + [101, 104, 103, 112],
        }
        self.df = pd.DataFrame(data)
        self.pad = pad
        self.strategy = SMCStrategy()

    def test_fvg_detection(self):
        # 第4根Low(108) > 第2根High(106)，產生FVG；第4根開盤 107 未超過前收 103 的 5% 跳空上限
        res_df = self.strategy.generate_signals(self.df, min_gap_pct=0.1, use_sma_filter=False)
        self.assertFalse(res_df['FVG'].iloc[self.pad + 2], "第三根不該有缺口")
        self.assertTrue(res_df['FVG'].iloc[self.pad + 3], "第四根必須判定出缺口")
        print("✅ 單元測試通過：FVG 缺口判定邏輯正常！")

if __name__ == '__main__':
    unittest.main()

# ==========================================
# 市場環境過濾 / 資料下載 回歸測試
# ==========================================
import os
import tempfile
from unittest import mock
import numpy as np

import regime
import portfolio_engine
import update_data


def _make_market(n=400, bull=True, vol=0.005):
    """產生合成大盤：bull=True 時穩定上漲 (站上 200MA)，vol 控制日波動。"""
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    drift = 0.001 if bull else -0.001
    rets = drift + rng.normal(0, vol, n)
    close = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({'Open': close, 'High': close * 1.01, 'Low': close * 0.99,
                         'Close': close, 'Volume': 1e7}, index=idx)


class TestRegime(unittest.TestCase):
    def test_classification(self):
        bull_low = regime.add_regime_columns(_make_market(bull=True, vol=0.003))
        self.assertEqual(bull_low['Regime'].iloc[-1], 'Bull_LowVol')
        self.assertEqual(bull_low['Regime'].iloc[0], 'Unknown')  # 均線尚未成形
        bear_high = regime.add_regime_columns(_make_market(bull=False, vol=0.03))
        self.assertTrue(bear_high['Regime'].iloc[-1].endswith('HighVol'))

    def test_portfolio_respects_allowed_regimes(self):
        market = _make_market(bull=True, vol=0.003)  # 後段全為 Bull_LowVol
        stock = market.copy()

        def fake_fetch(ticker, days):
            return market.copy() if ticker == "0050.TW" else stock.copy()

        entry_date = market.index[300]
        fake_trades = pd.DataFrame([{
            '日期': entry_date, '出場日期': market.index[310], '進場': 100.0, '出場': 110.0,
            '真實R': 2.0, '絕對損益': 1000.0, '每股風險': 5.0, '標的': '1234.TW', '板塊': 'X',
            'Alpha動能分數': 1.0, 'Volume': 1e7,
        }])

        class FakeExecutor:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def submit(self, fn, *a, **k):
                f = mock.Mock()
                f.result.return_value = fake_trades.copy() if fn is portfolio_engine._pure_calc_worker else fn(*a, **k)
                return f

        common = dict(tickers=['1234.TW'], years=2, bt_mode='固定盈虧比', bt_val=3.0, gap_pct=0.3,
                      use_sma=True, slippage_pct=0.002, fee_disc=0.28, min_vol=0,
                      sector_map={'1234.TW': 'X'})
        with mock.patch.object(portfolio_engine, 'fetch_stock_data', fake_fetch), \
             mock.patch.object(portfolio_engine, 'ThreadPoolExecutor', FakeExecutor), \
             mock.patch.object(portfolio_engine, 'ProcessPoolExecutor', FakeExecutor), \
             mock.patch.object(portfolio_engine, 'as_completed', lambda fs: fs):
            df_ok, stats_ok, _ = portfolio_engine.run_portfolio_backtest(
                allowed_regimes=('Bull_LowVol',), **common)
            df_blocked, stats_blocked, _ = portfolio_engine.run_portfolio_backtest(
                allowed_regimes=('Bear_HighVol',), **common)
            df_nofilter, _, _ = portfolio_engine.run_portfolio_backtest(allowed_regimes=None, **common)
            # 考試期只計 HOLDOUT_START 之後的訊號；這筆交易在 2021 年，應被排除
            df_holdout, _, _ = portfolio_engine.run_portfolio_backtest(
                allowed_regimes=None, period='holdout', **common)

        self.assertEqual(len(df_ok), 1)
        self.assertEqual(stats_ok['市場環境過濾淘汰次數'], 0)
        self.assertTrue(df_blocked.empty, "不在允許環境內的訊號不應進場")
        self.assertEqual(len(df_nofilter), 1)
        self.assertTrue(df_holdout.empty)


class TestUpdateData(unittest.TestCase):
    def test_no_double_suffix_and_matching_filename(self):
        idx = pd.date_range("2024-01-01", periods=20, tz="Asia/Taipei")
        hist = pd.DataFrame({'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Close': 1.0,
                             'Volume': 1, 'Dividends': 0, 'Stock Splits': 0}, index=idx)
        requested = []

        def fake_ticker(sym):
            requested.append(sym)
            t = mock.Mock()
            t.history.return_value = hist.copy()
            return t

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(update_data, 'DATA_DIR', d), \
             mock.patch.object(update_data, 'load_data', lambda: ({'半導體業': ['2330.TW']}, {}, [])), \
             mock.patch.object(update_data.yf, 'Ticker', fake_ticker), \
             mock.patch.object(update_data.time, 'sleep', lambda s: None):
            update_data.download_all_stocks(years=1)
            self.assertEqual(requested, ['2330.TW'])
            self.assertEqual(os.listdir(d), ['2330.parquet'])
            saved = pd.read_parquet(os.path.join(d, '2330.parquet'))
            self.assertEqual(list(saved.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])
            self.assertIsNone(saved.index.tz)


# ==========================================
# 回測真實度選項
# ==========================================
from backtest_engine import InstitutionalBacktestEngine, REALISM_DEFAULTS
from strategy import randomize_signals


def _bars(rows, signal_at=(), risk=3.0):
    """rows: [(O, H, L, C), ...]；signal_at: 哪幾根 K 棒有訊號。"""
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    df = pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close'], index=idx)
    df['FVG'] = [i in signal_at for i in range(len(rows))]
    df['adjusted_risk'] = risk
    return df


def _run(df, mode="固定盈虧比", rr=3.0, **opts):
    eng = InstitutionalBacktestEngine(fee_rate=0, tax_rate=0, slippage_pct=0, rr_ratio=rr, bt_mode=mode, **opts)
    trades, _, _ = eng.run_backtest(df.copy())
    return trades


class TestRealism(unittest.TestCase):
    def test_next_stop_fills_at_max_open_trigger(self):
        df = _bars([(100, 101, 99, 100), (102, 104, 101, 103), (103, 103, 90, 95)], signal_at=[0])
        t = _run(df, entry_mode="next_stop")
        self.assertEqual(len(t), 1)
        self.assertEqual(t['entry_price'].iloc[0], 102)        # 跳空高開，成交在開盤
        self.assertEqual(t['signal_time'].iloc[0], df.index[0])
        self.assertEqual(t['entry_time'].iloc[0], df.index[1])
        # 停損價在掛單時決定：觸價 100 − 風險 3 = 97，高開成交後實際虧損 > 1R
        self.assertEqual(t['exit_price'].iloc[0], 97)
        self.assertLess(t['real_r'].iloc[0], -1.0)

    def test_next_stop_cancelled_if_not_triggered(self):
        df = _bars([(100, 101, 99, 100), (99, 99.5, 97, 98), (98, 120, 98, 119)], signal_at=[0])
        self.assertTrue(_run(df, entry_mode="next_stop", close_open_at_end=True).empty)

    def test_skip_limit_up(self):
        rows = [(100, 100, 100, 100), (105, 110, 105, 110), (110, 111, 100, 101)]
        df = _bars(rows, signal_at=[1])
        self.assertEqual(len(_run(df)), 1)                       # 原始：漲停收盤也買得到
        self.assertTrue(_run(df, skip_limit_up=True).empty)     # 真實：漲停買不到
        # next_stop：成交日一字漲停買不到
        rows2 = [(100, 101, 99, 100), (110, 110, 110, 110), (110, 111, 100, 101)]
        self.assertTrue(_run(_bars(rows2, signal_at=[0]), entry_mode="next_stop", skip_limit_up=True).empty)

    def test_limit_down_lock(self):
        rows = [(100, 101, 99, 100), (90, 90, 90, 90), (91, 92, 85, 86)]
        df = _bars(rows, signal_at=[0])
        self.assertEqual(_run(df)['exit_time'].iloc[0], df.index[1])
        t = _run(df, limit_down_lock=True)
        self.assertEqual(t['exit_time'].iloc[0], df.index[2])  # 一字跌停隔天才賣得掉
        self.assertEqual(t['reason'].iloc[0], "Gap_Through_Stop")

    def test_protection_period_uses_emergency_stop(self):
        # 進場 100、風險 3：一般停損 97，緊急停損 94
        rows = [(100, 101, 99, 100), (99, 100, 96, 99), (99, 100, 93, 95)] + [(95, 96, 94.5, 95)] * 3
        df = _bars(rows, signal_at=[0])
        self.assertEqual(_run(df)['exit_time'].iloc[0], df.index[1])
        t = _run(df, protection_days=10)
        self.assertEqual(t['exit_time'].iloc[0], df.index[2])
        self.assertEqual(t['exit_price'].iloc[0], 94)
        self.assertAlmostEqual(t['real_r'].iloc[0], -2.0, places=2)
        # 保護期結束後恢復一般停損
        rows2 = [(100, 101, 99, 100)] + [(99, 100, 98, 99)] * 3 + [(98, 99, 96, 97)]
        t2 = _run(_bars(rows2, signal_at=[0]), protection_days=3)
        self.assertEqual(t2['reason'].iloc[0], "Hit_Stop")

    def test_one_position_per_ticker(self):
        rows = [(100, 101, 99, 100)] * 5 + [(100, 101, 90, 91)]
        df = _bars(rows, signal_at=[0, 1, 2])
        self.assertEqual(len(_run(df)), 3)
        self.assertEqual(len(_run(df, one_position_per_ticker=True)), 1)

    def test_realistic_intrabar_ordering(self):
        # 進場 100、風險 1、移動停利 2R：第 2 根先衝到 110 再跌到 105
        rows = [(100, 100, 100, 100), (101, 110, 105, 106), (106, 107, 100, 101)]
        df = _bars(rows, signal_at=[0], risk=1.0)
        orig = _run(df, mode="移動停利", rr=2.0)
        self.assertEqual(orig['exit_time'].iloc[0], df.index[1])  # 原始：同根 K 棒先用高點上調再被打到
        real = _run(df, mode="移動停利", rr=2.0, realistic_intrabar=True)
        self.assertEqual(real['exit_time'].iloc[0], df.index[2])
        self.assertEqual(real['exit_price'].iloc[0], 106)  # 移動停利 108，隔天開 106 跳空跌破
        self.assertEqual(real['reason'].iloc[0], "Gap_Through_Stop")

    def test_close_open_at_end(self):
        df = _bars([(100, 101, 99, 100), (100, 102, 99, 101)], signal_at=[0])
        self.assertTrue(_run(df).empty)
        t = _run(df, close_open_at_end=True)
        self.assertEqual(t['reason'].iloc[0], "End_Of_Data")

    def test_randomize_signals(self):
        df = _make_market(n=300)
        df['FVG'] = False
        df.iloc[[100, 150, 200], df.columns.get_loc('FVG')] = True
        a = randomize_signals(df, seed=1, salt="2330.TW")
        b = randomize_signals(df, seed=1, salt="2330.TW")
        c = randomize_signals(df, seed=2, salt="2330.TW")
        self.assertEqual(a['FVG'].sum(), 3)
        self.assertTrue((a['FVG'] == b['FVG']).all())
        self.assertFalse((a['FVG'] == c['FVG']).all())
        self.assertTrue((a['adjusted_risk'] == a['Close'] * 0.03).all())


class TestDataProvider(unittest.TestCase):
    def test_tz_aware_cache_is_normalized(self):
        import data_provider
        idx = pd.date_range("2024-01-01", periods=5, tz="Asia/Taipei")
        df = pd.DataFrame({'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Close': 1.0, 'Volume': 1}, index=idx)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                os.makedirs("local_data")
                df.to_parquet("local_data/9999.parquet")
                out = data_provider.fetch_stock_data("9999.TW", days=30)
            finally:
                os.chdir(cwd)
        self.assertIsNone(out.index.tz)
        self.assertEqual(len(out[out.index < pd.Timestamp("2024-01-03")]), 2)


class TestEntryVariants(unittest.TestCase):
    def test_breakout20(self):
        n = 100
        close = [100.0] * 90 + [101, 102, 103, 99, 98, 110, 100, 100, 100, 100]
        df = pd.DataFrame({'Open': close, 'High': [c + 0.5 for c in close],
                           'Low': [c - 0.5 for c in close], 'Close': close})
        out = SMCStrategy().generate_signals(df.copy(), entry_type='breakout20', use_sma_filter=False)
        # 收盤 > 前 20 日最高價 才算突破：101/102/103/110 是突破，99、98 不是
        self.assertEqual(list(out.index[out['FVG']]), [90, 91, 92, 95])
        self.assertTrue((out['adjusted_risk'] == out['Close'] * 0.03).all())

    def test_random_respects_mask(self):
        df = _make_market(n=300)
        df['FVG'] = False
        df.iloc[[100, 150, 200, 250], df.columns.get_loc('FVG')] = True
        mask = pd.Series(False, index=df.index)
        mask.iloc[200:220] = True
        out = randomize_signals(df, seed=3, salt="X", candidate_mask=mask)
        picked = np.flatnonzero(out['FVG'].values)
        self.assertEqual(len(picked), 4)
        self.assertTrue(((picked >= 200) & (picked < 220)).all())


class TestRetestEntry(unittest.TestCase):
    def _df(self, rows, top=97.0):
        df = _bars(rows, signal_at=[0])
        df['fvg_top'] = top
        return df

    def test_fills_on_pullback_at_limit(self):
        # 訊號日缺口上緣 97、風險 3 → 停損 94
        rows = [(100, 101, 97, 100), (100, 101, 98, 99), (98, 99, 96.5, 98), (98, 99, 93, 94)]
        t = _run(self._df(rows), entry_mode="retest")
        self.assertEqual(t['entry_time'].iloc[0], self._df(rows).index[2])
        self.assertEqual(t['entry_price'].iloc[0], 97)
        self.assertEqual(t['exit_price'].iloc[0], 94)
        self.assertAlmostEqual(t['real_r'].iloc[0], -1.0, places=2)

    def test_gap_down_fill_at_open(self):
        rows = [(100, 101, 97, 100), (96, 97, 95.5, 96), (96, 97, 93, 94)]
        t = _run(self._df(rows), entry_mode="retest")
        self.assertEqual(t['entry_price'].iloc[0], 96)  # 開低於限價，以開盤成交

    def test_cancel_when_open_below_stop_or_expired(self):
        rows = [(100, 101, 97, 100), (93, 94, 90, 92), (92, 99, 91, 98)]
        self.assertTrue(_run(self._df(rows), entry_mode="retest", close_open_at_end=True).empty)
        rows2 = [(100, 101, 97, 100)] + [(101, 102, 99, 101)] * 3 + [(98, 99, 96, 97)]
        self.assertTrue(_run(self._df(rows2), entry_mode="retest", retest_days=3, close_open_at_end=True).empty)
        self.assertEqual(len(_run(self._df(rows2), entry_mode="retest", retest_days=5, close_open_at_end=True)), 1)


# ==========================================
# 統計、逐日盯市回撤、Walk-Forward、消融實驗 回歸測試
# ==========================================
import data_provider
import walk_forward
from backtest_engine import run_backtest_core, summarize_trades
from research_pipeline import QuantResearchPipeline


def _random_walk(n=1600, seed=1):
    """合成個股 K 線：帶一點向上漂移的隨機漫步，足夠產生一些 FVG 訊號。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close,
                         'Volume': rng.integers(1_000_000, 5_000_000, n).astype(float)}, index=idx)


class TestSummarizeTrades(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(summarize_trades(pd.DataFrame())['trades'], 0)
        self.assertEqual(summarize_trades(None)['expectancy'], 0.0)

    def test_numbers(self):
        s = summarize_trades(pd.DataFrame({'真實R': [2.0, -1.0, -1.0, 4.0]}))
        self.assertEqual(s['trades'], 4)
        self.assertAlmostEqual(s['win_rate'], 0.5)
        self.assertAlmostEqual(s['expectancy'], 1.0)
        self.assertAlmostEqual(s['total_r'], 4.0)
        self.assertAlmostEqual(s['max_dd_r'], 2.0)   # 2 → 1 → 0：從高點回落 2R

    def test_core_returns_two_values_and_rejects_percent_slippage(self):
        df = _bars([(100, 101, 99, 100), (100, 110, 99, 109), (109, 110, 108, 109)], signal_at=[0])
        res = run_backtest_core(df.copy(), "固定盈虧比", 3.0, 0.002)
        self.assertEqual(len(res), 2)
        with self.assertRaises(ValueError):
            run_backtest_core(df.copy(), "固定盈虧比", 3.0, 0.1)   # 0.1 = 10%，一定是把百分比當比例傳


class TestMarkToMarket(unittest.TestCase):
    def test_open_loss_shows_up_in_equity(self):
        idx = pd.bdate_range("2024-01-01", periods=5)
        close = pd.Series([100.0, 80.0, 70.0, 95.0, 110.0], index=idx)
        pos = [{'ticker': 'A', 'entry_date': idx[0], 'exit_date': idx[4], 'shares': 100,
                'entry_price': 100.0, 'net_profit': 1000.0}]
        eq = portfolio_engine.mark_to_market_equity(pos, {'A': close}, 100000.0)
        self.assertAlmostEqual(eq[idx[0]], 100000.0)
        self.assertAlmostEqual(eq[idx[2]], 97000.0)      # 持倉中浮虧 30 × 100 股
        self.assertAlmostEqual(eq[idx[4]], 101000.0)     # 出場後只剩已實現損益
        dd = ((eq.cummax() - eq) / eq.cummax()).max()
        self.assertAlmostEqual(dd, 0.03)                 # 只看已實現損益的話回撤會是 0

    def test_missing_price_falls_back_to_realized(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        pos = [{'ticker': 'X', 'entry_date': idx[0], 'exit_date': idx[2], 'shares': 10,
                'entry_price': 50.0, 'net_profit': -200.0}]
        eq = portfolio_engine.mark_to_market_equity(pos, {}, 1000.0)
        self.assertAlmostEqual(eq.iloc[0], 1000.0)
        self.assertAlmostEqual(eq.iloc[-1], 800.0)
        self.assertTrue(portfolio_engine.mark_to_market_equity([], {}, 1000.0).empty)


class TestWalkForward(unittest.TestCase):
    def test_runs_and_returns_three_values(self):
        df = _random_walk()
        trades, logs, stats = walk_forward.run_wfo(df, train_years=2, test_years=1, use_sma_filter=False)
        self.assertEqual(len(logs), 4)   # 1600 根：(1600 − 504) // 252 = 4 段盲測
        if stats is not None:
            self.assertEqual(stats['總交易次數'], len(trades))
            # 盲測交易的訊號日一定落在第一段訓練期之後
            self.assertTrue((pd.to_datetime(trades['訊號日']) >= df.index[504]).all())

    def test_not_enough_data(self):
        trades, logs, stats = walk_forward.run_wfo(_random_walk(n=300))
        self.assertTrue(trades.empty)
        self.assertIsNone(stats)


class TestResearchPipeline(unittest.TestCase):
    def test_ablation_switches_actually_change_signals(self):
        report = QuantResearchPipeline(_random_walk()).run_ablation_study({'min_gap_pct': 0.3})
        self.assertEqual(len(report), 16)
        # 濾網全開的訊號數一定少於全關；如果參數名寫錯被策略忽略，兩者會一樣
        all_on = report[report[['use_sma_filter', 'use_bos_filter', 'use_vol_filter', 'use_atr_filter']].all(axis=1)]
        all_off = report[~report[['use_sma_filter', 'use_bos_filter', 'use_vol_filter', 'use_atr_filter']].any(axis=1)]
        self.assertLess(all_on['訊號數'].iloc[0], all_off['訊號數'].iloc[0])
        self.assertGreater(report['訊號數'].nunique(), 4)


class TestDataProvider(unittest.TestCase):
    def test_split_ticker(self):
        self.assertEqual(data_provider.split_ticker("2330.TW"), ("2330", "TW"))
        self.assertEqual(data_provider.split_ticker("6488.two"), ("6488", "TWO"))
        self.assertEqual(data_provider.split_ticker("2330"), ("2330", "TW"))

    def test_latest_local_date(self):
        idx = pd.bdate_range("2024-01-01", periods=5)
        df = pd.DataFrame({'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Close': 1.0, 'Volume': 1.0}, index=idx)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                self.assertIsNone(data_provider.latest_local_date("9999.TW"))
                os.makedirs("local_data")
                df.to_parquet("local_data/9999.parquet")
                self.assertEqual(data_provider.latest_local_date("9999.TW"), idx[-1])
            finally:
                os.chdir(cwd)

class TestVolSurgeFilter(unittest.TestCase):
    def test_max_vol_surge_removes_only_high_volume_signals(self):
        df = _random_walk(seed=3)
        base = dict(gap_pct=0.3, use_sma=False, bt_mode="移動停利", bt_val=10.0, realistic=True)
        all_trades = portfolio_engine._pure_calc_worker("T.TW", df.copy(), **base)
        capped = portfolio_engine._pure_calc_worker("T.TW", df.copy(), max_vol_surge=1.2, **base)
        self.assertGreater((all_trades['Alpha動能分數'] > 1.2).sum(), 0, "測試資料應該要有爆量訊號")
        # 前 20 根 K 棒還沒有 20 日均量 (爆量倍數為 NaN)，那些訊號不會被過濾，所以用「沒有任何一筆超過」來檢查
        self.assertFalse((capped['Alpha動能分數'] > 1.2).any())