import sys
import types

import numpy as np
import pandas as pd

from research.vcp_sector import data_loader as dl
from research.vcp_sector.config import DataConfig


def _cfg(tmp_path):
    return DataConfig(local_data_dir=tmp_path / "local_data", cache_dir=tmp_path / "cache",
                      start_date="2024-01-01")


def _fake_yf(available: set[str]):
    """假的 yfinance：只有 available 裡的 symbol 有資料，回傳 group_by='ticker' 的 MultiIndex。"""
    calls = []

    def download(symbols, **kw):
        calls.append(list(symbols))
        idx = pd.bdate_range("2024-01-01", periods=5)
        parts = {}
        for s in symbols:
            if s in available:
                parts[s] = pd.DataFrame({"Open": 10.0, "High": 11.0, "Low": 9.0,
                                         "Close": np.arange(10, 15, dtype=float), "Volume": 1000.0},
                                        index=idx)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, axis=1)

    mod = types.SimpleNamespace(download=download)
    return mod, calls


def test_yfinance_tw_then_two_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "last_complete_session", lambda **k: pd.Timestamp("2024-01-05"))
    mod, calls = _fake_yf({"2330.TW", "6488.TWO"})
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    monkeypatch.setattr(dl, "_finmind_get", lambda *a, **k: pd.DataFrame())
    cfg = _cfg(tmp_path)

    df = dl.load_prices(["2330", "6488"], cfg)
    assert set(df["stock_id"]) == {"2330", "6488"}
    assert calls[0] == ["2330.TW", "6488.TW"] and calls[1] == ["6488.TWO"]
    assert (cfg.cache_dir / "prices_adj" / "6488.parquet").exists()

    # 第二次只讀快取，不再下載
    calls.clear()
    df2 = dl.load_prices(["2330", "6488"], cfg)
    assert calls == [] and len(df2) == len(df)


def test_finmind_adj_denied_called_once_and_no_unadjusted(tmp_path, monkeypatch):
    mod, _ = _fake_yf(set())
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    monkeypatch.setattr(dl, "_FINMIND_ADJ_DENIED", False)
    seen = []

    def fake_get(dataset, cfg, **params):
        seen.append(dataset)
        out = pd.DataFrame()
        out.attrs["denied"] = True
        return out

    monkeypatch.setattr(dl, "_finmind_get", fake_get)
    df = dl.load_prices(["1101", "1102", "1103"], _cfg(tmp_path))
    assert df.empty
    assert seen == ["TaiwanStockPriceAdj"]          # 被拒一次就停
    assert "TaiwanStockPrice" not in seen          # 絕不退回未還原價


def test_old_unadjusted_cache_ignored(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    old = cfg.cache_dir / "prices"
    old.mkdir(parents=True)
    pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=3), "open": 1.0, "high": 1.0,
                  "low": 1.0, "close": 1.0, "volume": 1.0, "turnover": 1.0}).to_parquet(old / "1101.parquet")
    assert dl.load_prices(["1101"], cfg, use_api=False).empty


def test_local_data_multiindex_columns(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.local_data_dir.mkdir(parents=True)
    idx = pd.bdate_range("2024-01-01", periods=3)
    raw = pd.DataFrame({("Close", "2330.TW"): [1.0, 2, 3], ("High", "2330.TW"): 3.0,
                        ("Low", "2330.TW"): 1.0, ("Open", "2330.TW"): 1.0,
                        ("Volume", "2330.TW"): 10.0}, index=idx)
    raw.to_parquet(cfg.local_data_dir / "2330.TW.parquet")
    df = dl.load_prices(["2330"], cfg, use_api=False)
    assert df["close"].tolist() == [1.0, 2.0, 3.0]
    assert df["turnover"].tolist() == [10.0, 20.0, 30.0]


def test_last_complete_session():
    tz = "Asia/Taipei"
    wed_noon = pd.Timestamp("2026-11-04 12:00", tz=tz)
    wed_eve = pd.Timestamp("2026-11-04 15:00", tz=tz)
    mon_morning = pd.Timestamp("2026-11-02 09:00", tz=tz)
    sat = pd.Timestamp("2026-11-07 16:00", tz=tz)
    assert dl.last_complete_session(wed_noon) == pd.Timestamp("2026-11-03")   # 盤中 → 昨天
    assert dl.last_complete_session(wed_eve) == pd.Timestamp("2026-11-04")
    assert dl.last_complete_session(mon_morning) == pd.Timestamp("2026-10-30")
    assert dl.last_complete_session(sat) == pd.Timestamp("2026-11-06")


def test_partial_bar_not_cached_and_stale_cache_refetched(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(dl, "last_complete_session", lambda **k: pd.Timestamp("2024-01-04"))
    mod, calls = _fake_yf({"2330.TW"})              # 假資料 2024-01-01 ~ 01-05
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    monkeypatch.setattr(dl, "_finmind_get", lambda *a, **k: pd.DataFrame())

    df = dl.load_prices(["2330"], cfg)
    assert df["date"].max() == pd.Timestamp("2024-01-04")      # 01-05 盤中 K 棒被丟掉
    cached = pd.read_parquet(cfg.cache_dir / "prices_adj" / "2330.parquet")
    assert cached["date"].max() == pd.Timestamp("2024-01-04")

    # 時間往後推很多天 → 快取過期 → 重新下載
    monkeypatch.setattr(dl, "last_complete_session", lambda **k: pd.Timestamp("2024-02-01"))
    calls.clear()
    dl.load_prices(["2330"], cfg)
    assert calls == [["2330.TW"]]


def test_stale_local_data_refreshed(tmp_path, monkeypatch):
    cfg = DataConfig(local_data_dir=tmp_path / "local_data", cache_dir=tmp_path / "cache",
                     start_date="2024-01-01", max_stale_bdays=1)
    cfg.local_data_dir.mkdir(parents=True)
    idx = pd.bdate_range("2024-01-01", periods=2)            # 停在 01-02
    pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
                 index=idx).to_parquet(cfg.local_data_dir / "2330.parquet")
    monkeypatch.setattr(dl, "last_complete_session", lambda **k: pd.Timestamp("2024-01-05"))
    mod, calls = _fake_yf({"2330.TW"})
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    monkeypatch.setattr(dl, "_finmind_get", lambda *a, **k: pd.DataFrame())
    df = dl.load_prices(["2330"], cfg)
    assert calls == [["2330.TW"]] and df["date"].max() == pd.Timestamp("2024-01-05")
    # 不連網時仍照用舊的 local_data
    (cfg.cache_dir / "prices_adj" / "2330.parquet").unlink()
    assert dl.load_prices(["2330"], cfg, use_api=False)["date"].max() == pd.Timestamp("2024-01-02")
