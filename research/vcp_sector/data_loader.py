"""Phase 1：資料層。

統一輸出長表格式 ``DataFrame[date, stock_id, open, high, low, close, volume, turnover]``。

讀取順序（每檔股票）：
1. 既有 SMC 快取 ``local_data/{stock_id}.parquet``（yfinance auto_adjust，已還原權息）
2. 本模組快取 ``research/vcp_sector/cache/prices_adj/{stock_id}.parquet``
3. yfinance 批次下載（auto_adjust，先 .TW 再 .TWO）
4. FinMind ``TaiwanStockPriceAdj``（需贊助會員；被拒後不再呼叫）
不使用未還原價。

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

# get_data.py 的產業代碼表沒有 35 以後的新分類，stock_list.csv 會出現「未分類 (35)」等
TWSE_NEW_CATEGORIES = {
    "未分類 (35)": "綠能環保", "未分類 (36)": "數位雲端",
    "未分類 (37)": "運動休閒", "未分類 (38)": "居家生活",
}
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
                out = pd.DataFrame()
                out.attrs["denied"] = "level" in str(payload.get("msg", "")).lower()
                return out  # 權限 / 額度問題，重試無用
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
            "industry_category": csv["category"].replace(TWSE_NEW_CATEGORIES),
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


def last_complete_session(now: pd.Timestamp | None = None, close_hhmm: str = "14:30") -> pd.Timestamp:
    """最後一個「已收盤」的交易日（台北時間）。收盤前今日的 K 棒尚未完成，不可使用。"""
    now = now if now is not None else pd.Timestamp.now(tz="Asia/Taipei")
    today = now.tz_localize(None).normalize() if now.tzinfo else now.normalize()
    hh, mm = map(int, close_hhmm.split(":"))
    done_today = (now.hour, now.minute) >= (hh, mm) and today.dayofweek < 5
    return today if done_today else (today - pd.offsets.BDay(1)).normalize()


def _drop_incomplete(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["date"] <= last_complete_session(close_hhmm=cfg.session_close)]


def _is_fresh(df: pd.DataFrame, cfg: DataConfig) -> bool:
    """快取最後日期是否夠新（容許 max_stale_bdays 個營業日，涵蓋國定假日）。"""
    target = last_complete_session(close_hhmm=cfg.session_close)
    return df["date"].max() >= target - pd.offsets.BDay(cfg.max_stale_bdays)


def _cache_path(cfg: DataConfig, stock_id: str):
    # 只快取「還原權息」價；舊版 cache/prices/ 可能混有未還原價，故改用新資料夾、不再讀取
    return cfg.cache_dir / "prices_adj" / f"{stock_id}.parquet"


def _read_cached(stock_id: str, cfg: DataConfig, require_fresh: bool = False) -> pd.DataFrame:
    """local_data（SMC 系統自行維護，照用）→ 本模組快取（過期則視為沒有，重新下載）。"""
    own = _cache_path(cfg, stock_id)
    for path in (cfg.local_data_dir / f"{stock_id}.parquet",
                 cfg.local_data_dir / f"{stock_id}.TW.parquet",
                 cfg.local_data_dir / f"{stock_id}.TWO.parquet",
                 own):
        if path.exists():
            try:
                df = _read_local_parquet(path)
                if df.empty or (path == own and require_fresh and not _is_fresh(df, cfg)):
                    continue
                df["stock_id"] = stock_id
                return _drop_incomplete(df[LONG_COLS], cfg)
            except Exception as e:  # noqa: BLE001
                logger.warning("讀取快取失敗 %s: %s", path, e)
    return pd.DataFrame(columns=LONG_COLS)


def _save_cache(df: pd.DataFrame, stock_id: str, cfg: DataConfig) -> None:
    df = _drop_incomplete(df, cfg)  # 盤中未完成的 K 棒絕不寫入快取
    out = _cache_path(cfg, stock_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns="stock_id").to_parquet(out, index=False)


def _yf_symbol(stock_id: str, suffix: str) -> str:
    return "^TWII" if stock_id == "TAIEX" else f"{stock_id}{suffix}"


def _yf_batch(stock_ids: list[str], suffix: str, cfg: DataConfig) -> dict[str, pd.DataFrame]:
    """yfinance 批次下載（auto_adjust=True，已還原權息），與既有 SMC local_data 同來源。"""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("未安裝 yfinance，略過")
        return {}
    # 上櫃股先試 .TW 必然失敗，yfinance 會印大量 ERROR，屬正常現象，靜音
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(stock_ids), cfg.yf_batch_size):
        chunk = stock_ids[i:i + cfg.yf_batch_size]
        syms = {_yf_symbol(s, suffix): s for s in chunk}
        try:
            raw = yf.download(list(syms), start=cfg.start_date, auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("yfinance 批次下載失敗: %s", e)
            continue
        for sym, sid in syms.items():
            try:
                sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            sub = sub.copy()
            sub.columns = [str(c).lower() for c in sub.columns]
            if not set(PRICE_COLS).issubset(sub.columns):
                continue
            sub = sub[PRICE_COLS].dropna(subset=["close"])
            if sub.empty:
                continue
            sub.index = pd.to_datetime(sub.index).tz_localize(None) if getattr(sub.index, "tz", None) \
                else pd.to_datetime(sub.index)
            sub.index.name = "date"
            sub = sub.reset_index()
            sub["turnover"] = sub["close"] * sub["volume"]
            sub["stock_id"] = sid
            out[sid] = sub[LONG_COLS]
        logger.info("yfinance %s：%d/%d 檔", suffix or "index", min(i + cfg.yf_batch_size, len(stock_ids)),
                    len(stock_ids))
    return out


_FINMIND_ADJ_DENIED = False


def _finmind_adj(stock_id: str, cfg: DataConfig) -> pd.DataFrame:
    """FinMind 還原價（需贊助會員）。第一次被拒後整個 session 不再嘗試，避免浪費額度。"""
    global _FINMIND_ADJ_DENIED
    if _FINMIND_ADJ_DENIED:
        return pd.DataFrame()
    raw = _finmind_get("TaiwanStockPriceAdj", cfg, data_id=stock_id, start_date=cfg.start_date)
    if raw.attrs.get("denied"):
        _FINMIND_ADJ_DENIED = True
        logger.warning("FinMind 帳號無 TaiwanStockPriceAdj 權限，之後改用 yfinance 結果，不再呼叫")
    return _normalize_finmind_price(raw) if not raw.empty else pd.DataFrame()


def load_prices(stock_ids: Iterable[str], cfg: DataConfig = DEFAULT_CONFIG.data,
                use_api: bool = True) -> pd.DataFrame:
    """多檔日 K（長表格式，還原權息價）。

    來源順序：local_data → 本模組快取 → yfinance（.TW → .TWO）→ FinMind 還原價。
    **不使用未還原價**：除權息日的跳空會被當成下跌，污染均線與報酬。
    """
    ids = [str(s) for s in dict.fromkeys(stock_ids)]
    got: dict[str, pd.DataFrame] = {}
    for s in ids:
        df = _read_cached(s, cfg, require_fresh=use_api)
        if not df.empty:
            got[s] = df
    n_cached = len(got)
    missing = [s for s in ids if s not in got]
    logger.info("價格資料：快取命中 %d 檔，需下載 %d 檔", n_cached, len(missing))

    if use_api and missing:
        for suffix in (".TW", ".TWO"):
            todo = [s for s in missing if s not in got]
            if not todo:
                break
            for s, df in _yf_batch(todo, suffix, cfg).items():
                got[s] = _drop_incomplete(df, cfg)
                _save_cache(df, s, cfg)
        for s in [s for s in missing if s not in got]:
            df = _finmind_adj(s, cfg)
            if df.empty:
                if _FINMIND_ADJ_DENIED:
                    break
                continue
            got[s] = _drop_incomplete(df, cfg)
            _save_cache(df, s, cfg)

    failed = [s for s in ids if s not in got]
    logger.info("價格資料：共取得 %d 檔（快取 %d、新下載 %d），失敗 %d 檔%s", len(got), n_cached,
                len(got) - n_cached, len(failed), f"，例如 {failed[:10]}" if failed else "")

    if got:
        last = pd.Series({k: v["date"].max() for k, v in got.items() if not v.empty})
        stale = last[last < last.max() - pd.Timedelta(days=10)]
        if len(stale):
            logger.warning("%d 檔資料最後日期落後 10 天以上（例如 %s），近期會被視為不可交易；"
                           "若是 local_data 太舊，請先跑 update_data.py", len(stale),
                           dict(stale.dt.strftime("%Y-%m-%d").head(5)))

    frames = [f for f in got.values() if not f.empty]
    if not frames:
        return pd.DataFrame(columns=LONG_COLS)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["date"] >= pd.Timestamp(cfg.start_date)]
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_stock_prices(stock_id: str, cfg: DataConfig = DEFAULT_CONFIG.data,
                      use_api: bool = True) -> pd.DataFrame:
    """單檔日 K（長表格式）。找不到資料回傳空 DataFrame。"""
    return load_prices([stock_id], cfg, use_api)


def load_benchmark(cfg: DataConfig = DEFAULT_CONFIG.data, use_api: bool = True) -> pd.Series:
    """大盤收盤價序列（index=date）。

    順序：快取 / yfinance ^TWII → FinMind TAIEX（指數不需還原，免費帳號可用）→ 0050。
    註：TAIEX 是價格指數（不含息），而個股用還原價，group_rs 會系統性偏高約殖利率幅度；
    對所有族群相同，不影響 group_rank 與 is_strong。
    """
    s = load_prices([cfg.benchmark_id], cfg, use_api)
    if s.empty and use_api:
        raw = _finmind_get("TaiwanStockPrice", cfg, data_id=cfg.benchmark_id, start_date=cfg.start_date)
        if not raw.empty:
            s = _drop_incomplete(_normalize_finmind_price(raw), cfg)
            _save_cache(s, cfg.benchmark_id, cfg)
    if s.empty:
        s = load_prices([cfg.benchmark_fallback_id], cfg, use_api)
        if not s.empty:
            logger.warning("大盤改用 %s 代替 %s", cfg.benchmark_fallback_id, cfg.benchmark_id)
    if s.empty:
        raise RuntimeError("找不到大盤資料（TAIEX / 0050）")
    out = s.set_index("date")["close"].sort_index()
    out.name = str(s["stock_id"].iloc[0])
    return out


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
