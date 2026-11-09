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

        self.assertEqual(len(df_ok), 1)
        self.assertEqual(stats_ok['市場環境過濾淘汰次數'], 0)
        self.assertTrue(df_blocked.empty, "不在允許環境內的訊號不應進場")
        self.assertEqual(len(df_nofilter), 1)


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
