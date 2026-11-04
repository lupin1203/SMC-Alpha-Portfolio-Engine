"""Phase 1：資料層。

統一輸出長表格式 ``DataFrame[date, stock_id, open, high, low, close, volume, turnover]``。

讀取順序（每檔股票）：
1. 既有 SMC 快取 ``local_data/{stock_id}.parquet``（yfinance auto_adjust，已還原權息）
2. 本模組快取 ``research/vcp_sector/cache/prices/{stock_id}.parquet``
3. FinMind API（``TaiwanStockPriceAdj``，失敗再退回未還原的 ``TaiwanStockPrice`` 並記 warning）

過濾（ETF / 上市天數 / 成交值）全部以「逐日 eligibility mask」表示，
只用當日（含）以前的資料，不會用到未來才知道的上市天數或成交值。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, DataConfig

logger = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close", "volume"]
LONG_COLS = ["date", "stock_id"] + PRICE_COLS + ["turnover"]


# ---------------------------------------------------------------------------
# FinMind
# ---------------------------------------------------------------------------
def _finmind_get(dataset: str, cfg: DataConfig, **params) -> pd.DataFrame:
    import requests

    query = {"dataset": dataset, **params}
    token = os.environ.get(cfg.finmind_token_env)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(3):
        try:
            resp = requests.get(cfg.finmind_url, params=query, headers=headers, timeout=30)
            payload = resp.json()
            if resp.status_code == 200 and payload.get("status", 200) == 200:
                return pd.DataFrame(payload.get("data", []))
            logger.warning("FinMind %s %s 回應 %s: %s", dataset, params, resp.status_code, payload.get("msg"))
            if resp.status_code in (400, 401, 402, 403):
                break  # 權限 / 額度問題，重試無用
        except Exception as e:  # noqa: BLE001
            logger.warning("FinMind %s 第 %d 次失敗: %s", dataset, attempt + 1, e)
        time.sleep(2 ** attempt)
    return pd.DataFrame()


def _normalize_finmind_price(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume",
                             "Trading_money": "turnover"})
    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    if "turnover" not in df:
        df["turnover"] = df["close"] * df["volume"]
    return df[LONG_COLS]


# ---------------------------------------------------------------------------
# 產業分類
# ---------------------------------------------------------------------------
def load_stock_info(cfg: DataConfig = DEFAULT_CONFIG.data, use_api: bool = True) -> pd.DataFrame:
    """回傳 ``DataFrame[stock_id, stock_name, industry_category]``，已剔除 ETF / 權證等。

    優先順序：本模組快取 → FinMind ``TaiwanStockInfo`` → 既有 ``stock_list.csv``。
    """
    cache = cfg.cache_dir / "stock_info.parquet"
    info = pd.DataFrame()
    if cache.exists():
        info = pd.read_parquet(cache)
    elif use_api:
        raw = _finmind_get("TaiwanStockInfo", cfg)
        if not raw.empty:
            info = raw[
                ["stock_id", "stock_name", "industry_category"]
            ].drop_duplicates("stock_id", keep="last")
            cache.parent.mkdir(parents=True, exist_ok=True)
            info.to_parquet(cache)

    if info.empty and cfg.stock_list_csv.exists():
        csv = pd.read_csv(cfg.stock_list_csv, encoding="utf-8-sig")
        info = pd.DataFrame({
            "stock_id": csv["ticker"].astype(str).str.split(".").str[0],
            "stock_name": csv["name"],
            "industry_category": csv["category"],
        })

    return filter_security_type(info, cfg)


def filter_security_type(info: pd.DataFrame, cfg: DataConfig = DEFAULT_CONFIG.data) -> pd.DataFrame:
    """剔除 ETF、ETN、權證、特別股、存託憑證等：只留 4 碼、非 00 開頭的普通股。"""
    if info.empty:
        return info
    sid = info["stock_id"].astype(str)
    is_common = sid.str.fullmatch(r"[1-9]\d{3}")
    bad_cat = info["industry_category"].isin(cfg.excluded_categories)
    return info[is_common & ~bad_cat].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 價格
# ---------------------------------------------------------------------------
def _read_local_parquet(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):  # yf.download 會產生 (field, ticker)
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    if not set(PRICE_COLS).issubset(df.columns):
        return pd.DataFrame()
    df = df[PRICE_COLS].copy()
    df["turnover"] = df["close"] * df["volume"]
    df.index.name = "date"
    return df.reset_index()


def load_stock_prices(stock_id: str, cfg: DataConfig = DEFAULT_CONFIG.data,
                      use_api: bool = True) -> pd.DataFrame:
    """單檔日 K（長表格式）。找不到資料回傳空 DataFrame。"""
    stock_id = str(stock_id)
    for path in (cfg.local_data_dir / f"{stock_id}.parquet",
                 cfg.local_data_dir / f"{stock_id}.TW.parquet",
                 cfg.cache_dir / "prices" / f"{stock_id}.parquet"):
        if path.exists():
            try:
                df = _read_local_parquet(path)
                if not df.empty:
                    df["stock_id"] = stock_id
                    return df[LONG_COLS]
            except Exception as e:  # noqa: BLE001
                logger.warning("讀取快取失敗 %s: %s", path, e)

    if not use_api:
        return pd.DataFrame(columns=LONG_COLS)

    raw = _finmind_get("TaiwanStockPriceAdj", cfg, data_id=stock_id, start_date=cfg.start_date)
    if raw.empty:
        raw = _finmind_get("TaiwanStockPrice", cfg, data_id=stock_id, start_date=cfg.start_date)
        if not raw.empty:
            logger.warning("%s 只取得未還原權息價（TaiwanStockPrice）", stock_id)
    if raw.empty:
        return pd.DataFrame(columns=LONG_COLS)

    df = _normalize_finmind_price(raw)
    out = cfg.cache_dir / "prices" / f"{stock_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns="stock_id").to_parquet(out, index=False)
    return df


def load_prices(stock_ids: Iterable[str], cfg: DataConfig = DEFAULT_CONFIG.data,
                use_api: bool = True) -> pd.DataFrame:
    frames = [load_stock_prices(s, cfg, use_api) for s in stock_ids]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=LONG_COLS)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["date"] >= pd.Timestamp(cfg.start_date)]
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_benchmark(cfg: DataConfig = DEFAULT_CONFIG.data, use_api: bool = True) -> pd.Series:
    """大盤收盤價序列（index=date）。TAIEX 優先，退回 0050。"""
    for sid in (cfg.benchmark_id, cfg.benchmark_fallback_id):
        df = load_stock_prices(sid, cfg, use_api)
        if not df.empty:
            if sid != cfg.benchmark_id:
                logger.warning("大盤改用 %s 代替 %s", sid, cfg.benchmark_id)
            s = df.set_index("date")["close"].sort_index()
            s.name = sid
            return s
    raise RuntimeError("找不到大盤資料（TAIEX / 0050）")


# ---------------------------------------------------------------------------
# 寬表 + point-in-time 過濾
# ---------------------------------------------------------------------------
def to_wide(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    """長表 → 寬表（index=date, columns=stock_id）。"""
    return prices.pivot(index="date", columns="stock_id", values=field).sort_index()


def eligibility_mask(prices: pd.DataFrame, cfg: DataConfig = DEFAULT_CONFIG.data) -> pd.DataFrame:
    """逐日可交易池：截至當日已上市 >= min_listed_days 且近 N 日均成交值 >= 門檻。

    全部只用當日（含）以前的資料，因此可直接用於回測，不會前視。
    """
    close = to_wide(prices, "close")
    turnover = to_wide(prices, "turnover")
    listed_days = close.notna().cumsum()
    avg_turnover = turnover.rolling(cfg.turnover_window, min_periods=cfg.turnover_window).mean()
    mask = (listed_days >= cfg.min_listed_days) & (avg_turnover >= cfg.min_avg_turnover) & close.notna()
    return mask.fillna(False)
