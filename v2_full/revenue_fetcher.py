# revenue_fetcher.py
"""
月營收 YoY 動能 + 毛利率質量篩選器
資料來源：FinMind API (TaiwanStockMonthRevenue / TaiwanStockFinancialStatements)
快取策略：本地 JSON，當月內不重複抓取
"""
import os
import json
import logging
import time
import datetime
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ==========================================
# 設定區
# ==========================================
FINMIND_URL   = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

CACHE_DIR          = "./revenue_cache"
REVENUE_CACHE_FILE = os.path.join(CACHE_DIR, "monthly_revenue.json")
MARGIN_CACHE_FILE  = os.path.join(CACHE_DIR, "gross_margin.json")
REVENUE_HISTORY_DIR = os.path.join(CACHE_DIR, "history")   # 歷史序列存這裡
MARGIN_HISTORY_DIR  = os.path.join(CACHE_DIR, "margin_history")
CASHFLOW_HISTORY_DIR = os.path.join(CACHE_DIR, "cashflow_history")  # 營業現金流歷史序列
DEBT_HISTORY_DIR     = os.path.join(CACHE_DIR, "debt_history")       # 負債比率歷史序列
CACHE_TTL_DAYS     = 1


# ==========================================
# 公告延遲計算（消除前視偏差）
# ==========================================
# 回測時，財報資料不能在「資料所屬期間」當下就採用，
# 必須延後到市場真正能取得的「法定公告日」才可用，否則就是前視偏差。
# 實盤掃描端不套用這些延遲（當天 FinMind 有資料就代表市場已知）。

def _revenue_available_date(period_ts: pd.Timestamp) -> pd.Timestamp:
    """
    月營收可用日期 = 所屬月份的「次月 10 日」。
    依台股法規，公司須在次月 10 日前公告上月營收，
    保守起見一律壓到次月 10 日（回測寧可低估資訊優勢，不可高估）。
    """
    period_ts = pd.Timestamp(period_ts)
    year  = period_ts.year
    month = period_ts.month
    # 次月
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1
    return pd.Timestamp(year=next_year, month=next_month, day=10)


def _margin_available_date(period_ts: pd.Timestamp) -> pd.Timestamp:
    """
    季報（毛利率）可用日期 = 該季法定公告期限。
    依台股法規分季處理：
      Q1（3 月底結帳）→ 5/15
      Q2（6 月底結帳）→ 8/14
      Q3（9 月底結帳）→ 11/14
      Q4 / 年報（12 月底結帳）→ 隔年 3/31
    FinMind 季報 date 多為季末日（3/31、6/30、9/30、12/31），
    用月份判斷屬於哪一季，回傳對應法定公告日。
    """
    period_ts = pd.Timestamp(period_ts)
    year  = period_ts.year
    month = period_ts.month

    if month <= 3:        # Q1
        return pd.Timestamp(year=year, month=5, day=15)
    elif month <= 6:      # Q2
        return pd.Timestamp(year=year, month=8, day=14)
    elif month <= 9:      # Q3
        return pd.Timestamp(year=year, month=11, day=14)
    else:                 # Q4 / 年報 → 隔年 3/31
        return pd.Timestamp(year=year + 1, month=3, day=31)


# ==========================================
# 快取讀寫工具
# ==========================================
def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _load_cache(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(path: str, data: dict):
    _ensure_cache_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _cache_is_fresh(cache: dict, key: str) -> bool:
    """確認快取的 key 是否在 TTL 內"""
    meta = cache.get("__meta__", {}).get(key, {})
    saved_date = meta.get("date", "")
    if not saved_date:
        return False
    delta = (datetime.date.today() - datetime.date.fromisoformat(saved_date)).days
    return delta < CACHE_TTL_DAYS


# ==========================================
# FinMind 通用抓取器
# ==========================================
def _finmind_get(dataset: str, data_id: str, start_date: str) -> pd.DataFrame:
    """呼叫 FinMind API，回傳 DataFrame；失敗回傳空 df"""
    params = {
        "dataset":    dataset,
        "data_id":    data_id,
        "start_date": start_date,
    }
    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN

    for attempt in range(3):
        try:
            resp = requests.get(FINMIND_URL, params=params, timeout=10)
            data = resp.json()
            if data.get("status") == 200 and data.get("data"):
                return pd.DataFrame(data["data"])
            logger.warning(f"FinMind [{dataset}] {data_id} 無資料: {data.get('msg', '')}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"FinMind 第 {attempt+1} 次失敗 ({data_id}): {e}")
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()


# ==========================================
# 台股指數抓取（Regime 判斷用）
# ==========================================
def fetch_taiwan_index(index_id: str = "TAIEX", days: int = 250) -> pd.DataFrame:
    """
    用 FinMind 抓台股指數，供 Regime 判斷用。
    index_id: "TAIEX"（加權指數）或 "TPEx"（櫃買指數）
    回傳含 OHLC 欄位的 DataFrame，index 為日期。
    """
    start_date = (datetime.date.today() - datetime.timedelta(days=days + 100)).strftime("%Y-%m-%d")
    df = _finmind_get("TaiwanStockTotalReturnIndex", index_id, start_date)

    if df.empty or "price" not in df.columns:
        return pd.DataFrame()

    df = df[["date", "price"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={"price": "Close"})
    for col in ["Open", "High", "Low"]:
        df[col] = df["Close"]
    df["Volume"] = 0
    return df


# ==========================================
# 歷史序列抓取（回測用）
# ==========================================

def fetch_revenue_history(ticker: str, start_year: int = 2010) -> pd.DataFrame:
    """
    抓取單一標的從 start_year 至今的完整月營收序列，
    計算每月的 YoY，存成本地 parquet 供回測使用。

    回傳格式：DatetimeIndex（每月公告日），columns: revenue, yoy
    本地路徑：./revenue_cache/history/{ticker}.parquet
    """
    _ensure_cache_dir()
    os.makedirs(REVENUE_HISTORY_DIR, exist_ok=True)

    local_path = os.path.join(REVENUE_HISTORY_DIR, f"{ticker}.parquet")

    # 月營收每月才更新一次，7 天內的快取直接用，不重打 API
    if os.path.exists(local_path):
        mtime = datetime.date.fromtimestamp(os.path.getmtime(local_path))
        if (datetime.date.today() - mtime).days < 7:
            try:
                return pd.read_parquet(local_path)
            except Exception:
                pass

    start_date = f"{start_year}-01-01"
    df = _finmind_get("TaiwanStockMonthRevenue", ticker, start_date)

    if df.empty or "revenue" not in df.columns:
        return pd.DataFrame()

    df["date"]    = pd.to_datetime(df["date"])
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.sort_values("date").drop_duplicates("date").dropna(subset=["revenue"])
    df = df.set_index("date")

    # YoY = (本月 - 去年同月) / 去年同月 * 100
    df["yoy"] = ((df["revenue"] - df["revenue"].shift(12)) /
                 df["revenue"].shift(12) * 100).round(2)

    # 基期過低保護：去年同月營收太小（趨近 0）會讓 YoY 炸成天文數字（假爆發）
    # 去年同月營收 < 1000 千元（百萬）視為基期異常，該月 YoY 設為無效
    base_revenue = df["revenue"].shift(12)
    df.loc[base_revenue < 1000, "yoy"] = pd.NA
    # YoY 絕對值超過 500% 也視為異常基期造成的雜訊，過濾掉
    df.loc[df["yoy"].abs() > 500, "yoy"] = pd.NA

    df = df[["revenue", "yoy"]].dropna(subset=["yoy"])

    # ✅ 消除前視偏差：計算每筆營收的「市場可用日期」（次月 10 日）
    #    回測時用此日期比對，而非營收所屬月份
    df["available_date"] = df.index.map(_revenue_available_date)

    try:
        df.to_parquet(local_path)
    except Exception as e:
        logger.warning(f"月營收歷史存檔失敗 {ticker}: {e}")

    return df


def fetch_margin_history(ticker: str, start_year: int = 2010) -> pd.DataFrame:
    """
    抓取單一標的從 start_year 至今的完整季毛利率序列，
    存成本地 parquet 供回測使用。

    回傳格式：DatetimeIndex（季報公告日），columns: gross_margin, expanding
    本地路徑：./revenue_cache/margin_history/{ticker}.parquet
    """
    import numpy as np
    _ensure_cache_dir()
    os.makedirs(MARGIN_HISTORY_DIR, exist_ok=True)

    local_path = os.path.join(MARGIN_HISTORY_DIR, f"{ticker}.parquet")

    # 季報每季才更新一次，7 天內的快取直接用
    if os.path.exists(local_path):
        mtime = datetime.date.fromtimestamp(os.path.getmtime(local_path))
        if (datetime.date.today() - mtime).days < 7:
            try:
                return pd.read_parquet(local_path)
            except Exception:
                pass

    start_date = f"{start_year}-01-01"
    df = _finmind_get("TaiwanStockFinancialStatements", ticker, start_date)

    if df.empty:
        return pd.DataFrame()

    # 找毛利率欄位
    gm_df = df[df.get("type", pd.Series(dtype=str)).str.contains(
        "毛利率|GrossMargin", case=False, na=False
    )].copy()

    if gm_df.empty:
        # 嘗試自己算
        gp = df[df.get("type", pd.Series(dtype=str)).str.contains("毛利$|GrossProfit", case=False, na=False)]
        rv = df[df.get("type", pd.Series(dtype=str)).str.contains("營業收入|Revenue", case=False, na=False)]
        if gp.empty or rv.empty:
            return pd.DataFrame()
        gp = gp[["date", "value"]].rename(columns={"value": "gp"})
        rv = rv[["date", "value"]].rename(columns={"value": "rv"})
        gm_df = gp.merge(rv, on="date")
        gm_df["gross_margin"] = (pd.to_numeric(gm_df["gp"], errors="coerce") /
                                  pd.to_numeric(gm_df["rv"], errors="coerce") * 100).round(2)
    else:
        gm_df["gross_margin"] = pd.to_numeric(gm_df["value"], errors="coerce")

    gm_df["date"] = pd.to_datetime(gm_df["date"])
    gm_df = gm_df.sort_values("date").drop_duplicates("date").dropna(subset=["gross_margin"])
    gm_df = gm_df.set_index("date")[["gross_margin"]]

    # expanding = 本季 > 前一季
    gm_df["expanding"] = gm_df["gross_margin"] > gm_df["gross_margin"].shift(1)

    # ✅ 消除前視偏差：計算每季毛利率的「市場可用日期」（法定公告日，分季處理）
    gm_df["available_date"] = gm_df.index.map(_margin_available_date)

    try:
        gm_df.to_parquet(local_path)
    except Exception as e:
        logger.warning(f"毛利率歷史存檔失敗 {ticker}: {e}")

    return gm_df


def fetch_cashflow_history(ticker: str, start_year: int = 2010) -> pd.DataFrame:
    """
    抓取單一標的從 start_year 至今的完整季營業現金流序列，
    存成本地 parquet 供回測使用。

    帳面賺錢但營業現金流為負，常是塞貨或做假帳的警訊（地雷股防禦）。

    回傳格式：DatetimeIndex（季報結算日），columns: ocf, ocf_positive, available_date
    本地路徑：./revenue_cache/cashflow_history/{ticker}.parquet

    ✅ 與毛利率同樣套用季報公告延遲（available_date），消除前視偏差：
       Q1→5/15、Q2→8/14、Q3→11/14、Q4→隔年3/31
    """
    _ensure_cache_dir()
    os.makedirs(CASHFLOW_HISTORY_DIR, exist_ok=True)

    local_path = os.path.join(CASHFLOW_HISTORY_DIR, f"{ticker}.parquet")

    # 季報每季才更新一次，7 天內的快取直接用
    if os.path.exists(local_path):
        mtime = datetime.date.fromtimestamp(os.path.getmtime(local_path))
        if (datetime.date.today() - mtime).days < 7:
            try:
                return pd.read_parquet(local_path)
            except Exception:
                pass

    start_date = f"{start_year}-01-01"
    df = _finmind_get("TaiwanStockCashFlowsStatement", ticker, start_date)

    if df.empty:
        return pd.DataFrame()

    # 找營業活動現金流欄位
    ocf_df = df[df.get("type", pd.Series(dtype=str)).str.contains(
        "營業活動|CashFlowsFromOperating|OperatingActivities", case=False, na=False
    )].copy()

    if ocf_df.empty:
        return pd.DataFrame()

    ocf_df["ocf"]  = pd.to_numeric(ocf_df["value"], errors="coerce")
    ocf_df["date"] = pd.to_datetime(ocf_df["date"])
    ocf_df = ocf_df.sort_values("date").drop_duplicates("date").dropna(subset=["ocf"])
    ocf_df = ocf_df.set_index("date")[["ocf"]]

    if ocf_df.empty:
        return pd.DataFrame()

    ocf_df["ocf_positive"] = ocf_df["ocf"] > 0

    # ✅ 消除前視偏差：季報法定公告日（與毛利率同一套延遲規則）
    ocf_df["available_date"] = ocf_df.index.map(_margin_available_date)

    try:
        ocf_df.to_parquet(local_path)
    except Exception as e:
        logger.warning(f"現金流歷史存檔失敗 {ticker}: {e}")

    return ocf_df


def fetch_debt_ratio_history(ticker: str, start_year: int = 2010) -> pd.DataFrame:
    """
    抓取單一標的從 start_year 至今的完整季負債比率序列。
    負債比率 = 總負債 / 總資產（資產負債表，每季更新）

    金融/保險/建設業天生高負債（60-90%），這是產業特性不是風險，
    進場過濾時建議門檻設 60-70%，正好把不適合 SMC 策略的這些板塊擋掉。

    回傳格式：DatetimeIndex（季報公告日），columns: debt_ratio, available_date
    本地路徑：./revenue_cache/debt_history/{ticker}.parquet
    """
    _ensure_cache_dir()
    os.makedirs(DEBT_HISTORY_DIR, exist_ok=True)

    local_path = os.path.join(DEBT_HISTORY_DIR, f"{ticker}.parquet")

    # 季報每季才更新一次，7 天內的快取直接用
    if os.path.exists(local_path):
        mtime = datetime.date.fromtimestamp(os.path.getmtime(local_path))
        if (datetime.date.today() - mtime).days < 7:
            try:
                return pd.read_parquet(local_path)
            except Exception:
                pass

    start_date = f"{start_year}-01-01"
    df = _finmind_get("TaiwanStockBalanceSheet", ticker, start_date)

    if df.empty:
        return pd.DataFrame()

    # 取總資產與總負債
    assets_df = df[df.get("type", pd.Series(dtype=str)).str.contains(
        "TotalAssets|資產總計|資產合計", case=False, na=False
    )].copy()
    liab_df = df[df.get("type", pd.Series(dtype=str)).str.contains(
        "TotalLiabilities|負債總計|負債合計", case=False, na=False
    )].copy()

    if assets_df.empty or liab_df.empty:
        return pd.DataFrame()

    assets_df["value"] = pd.to_numeric(assets_df["value"], errors="coerce")
    liab_df["value"]   = pd.to_numeric(liab_df["value"],   errors="coerce")

    assets_df["date"] = pd.to_datetime(assets_df["date"])
    liab_df["date"]   = pd.to_datetime(liab_df["date"])

    assets_s = assets_df.groupby("date")["value"].sum()
    liab_s   = liab_df.groupby("date")["value"].sum()

    merged = pd.DataFrame({"total_assets": assets_s, "total_liabilities": liab_s}).dropna()
    if merged.empty:
        return pd.DataFrame()

    merged["debt_ratio"] = (merged["total_liabilities"] / merged["total_assets"] * 100).round(2)
    merged = merged[["debt_ratio"]].copy()

    # ✅ 消除前視偏差：套用季報公告延遲（與毛利率相同規則）
    merged["available_date"] = merged.index.map(_margin_available_date)

    try:
        merged.to_parquet(local_path)
    except Exception as e:
        logger.warning(f"負債比率歷史存檔失敗 {ticker}: {e}")

    return merged


def get_fundamental_at_date(
    ticker: str,
    query_date,
    rev_history: pd.DataFrame = None,
    margin_history: pd.DataFrame = None,
    cashflow_history: pd.DataFrame = None,
    debt_history: pd.DataFrame = None,
) -> dict:
    """
    查詢某個時間點的基本面狀態（用歷史序列，無前視偏差）。

    ✅ 關鍵：用「市場可用日期」(available_date) 比對，而非資料所屬期間。
       月營收：取 available_date（次月10日）<= query_date 的最近一筆
       毛利率：取 available_date（法定公告日）<= query_date 的最近一季
       負債比率：取 available_date（法定公告日）<= query_date 的最近一季
       這確保回測在某天只能看到「那天市場真正已公告」的財報，消除前視偏差。

    向後相容：若 parquet 沒有 available_date 欄位（舊快取），
             即時用所屬期間推算，避免報錯。

    回傳：{"latest_yoy": float|None, "gm_expanding": bool|None,
           "ocf_positive": bool|None, "debt_ratio": float|None}
    """
    query_ts = pd.Timestamp(query_date)
    result   = {"latest_yoy": None, "gm_expanding": None, "ocf_positive": None, "debt_ratio": None}

    # 月營收 YoY
    if rev_history is not None and not rev_history.empty:
        if "available_date" in rev_history.columns:
            avail = rev_history["available_date"]
            if not pd.api.types.is_datetime64_any_dtype(avail):
                avail = pd.to_datetime(avail)
        else:
            # 舊快取沒有此欄位 → 即時推算（次月10日）
            avail = pd.Series(rev_history.index.map(_revenue_available_date), index=rev_history.index)
        mask = avail.values <= query_ts.to_datetime64()
        if mask.any():
            result["latest_yoy"] = float(rev_history["yoy"].values[mask][-1])

    # 毛利率擴張
    if margin_history is not None and not margin_history.empty:
        if "available_date" in margin_history.columns:
            avail_gm = margin_history["available_date"]
            if not pd.api.types.is_datetime64_any_dtype(avail_gm):
                avail_gm = pd.to_datetime(avail_gm)
        else:
            avail_gm = pd.Series(margin_history.index.map(_margin_available_date), index=margin_history.index)
        mask_gm = avail_gm.values <= query_ts.to_datetime64()
        if mask_gm.any():
            result["gm_expanding"] = bool(margin_history["expanding"].values[mask_gm][-1])

    # 營業現金流是否為正（用 available_date 比對，無前視偏差）
    if cashflow_history is not None and not cashflow_history.empty:
        if "available_date" in cashflow_history.columns:
            avail_cf = cashflow_history["available_date"]
            if not pd.api.types.is_datetime64_any_dtype(avail_cf):
                avail_cf = pd.to_datetime(avail_cf)
        else:
            avail_cf = pd.Series(cashflow_history.index.map(_margin_available_date), index=cashflow_history.index)
        mask_cf = avail_cf.values <= query_ts.to_datetime64()
        if mask_cf.any():
            result["ocf_positive"] = bool(cashflow_history["ocf_positive"].values[mask_cf][-1])

    # 負債比率（用 available_date 比對，無前視偏差）
    if debt_history is not None and not debt_history.empty:
        if "available_date" in debt_history.columns:
            avail_dr = debt_history["available_date"]
            if not pd.api.types.is_datetime64_any_dtype(avail_dr):
                avail_dr = pd.to_datetime(avail_dr)
        else:
            avail_dr = pd.Series(debt_history.index.map(_margin_available_date), index=debt_history.index)
        mask_dr = avail_dr.values <= query_ts.to_datetime64()
        if mask_dr.any():
            result["debt_ratio"] = float(debt_history["debt_ratio"].values[mask_dr][-1])

    return result


def build_fundamental_history_cache(tickers: list, delay: float = 0.3) -> dict:
    """
    批量預載一組標的的完整歷史基本面序列，
    回傳 {ticker: {"rev": DataFrame, "margin": DataFrame}} 供回測引擎使用。
    7 天內的本地快取直接讀，不打 API（避免被 FinMind 封 IP）。
    內建限流斷路器：連續多次抓到空資料就停止打 API，剩下全用本地快取。
    """
    import os
    cache = {}
    api_calls = 0
    cache_hits = 0
    consecutive_empty = 0   # 連續抓到空資料的次數
    api_disabled = False    # 斷路器：觸發後不再打 API
    CIRCUIT_BREAK_THRESHOLD = 10  # 連續 10 次空資料就斷路

    for i, raw_ticker in enumerate(tickers):
        ticker = str(raw_ticker).replace(".TW", "")
        try:
            rev_path    = os.path.join(REVENUE_HISTORY_DIR, f"{ticker}.parquet")
            margin_path = os.path.join(MARGIN_HISTORY_DIR,  f"{ticker}.parquet")
            cashflow_path = os.path.join(CASHFLOW_HISTORY_DIR, f"{ticker}.parquet")
            debt_path   = os.path.join(DEBT_HISTORY_DIR, f"{ticker}.parquet")
            has_local = os.path.exists(rev_path)
            needs_api = (
                not has_local or
                (datetime.date.today() - datetime.date.fromtimestamp(os.path.getmtime(rev_path))).days >= 7
            )

            # 斷路器啟動後：有本地就讀本地，沒有就給空，一律不打 API
            if api_disabled and needs_api:
                if has_local:
                    try:
                        rev = pd.read_parquet(rev_path)
                        margin = pd.read_parquet(margin_path) if os.path.exists(margin_path) else pd.DataFrame()
                        cashflow = pd.read_parquet(cashflow_path) if os.path.exists(cashflow_path) else pd.DataFrame()
                        debt = pd.read_parquet(debt_path) if os.path.exists(debt_path) else pd.DataFrame()
                    except Exception:
                        rev, margin, cashflow, debt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                else:
                    rev, margin, cashflow, debt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                cache[raw_ticker] = {"rev": rev, "margin": margin, "cashflow": cashflow, "debt": debt}
                continue

            rev    = fetch_revenue_history(ticker)
            margin = fetch_margin_history(ticker)
            cashflow = fetch_cashflow_history(ticker)
            debt = fetch_debt_ratio_history(ticker)
            cache[raw_ticker] = {"rev": rev, "margin": margin, "cashflow": cashflow, "debt": debt}

            if needs_api:
                api_calls += 1
                # 偵測限流：打了 API 卻拿到空資料
                if rev is None or rev.empty:
                    consecutive_empty += 1
                    if consecutive_empty >= CIRCUIT_BREAK_THRESHOLD:
                        api_disabled = True
                        logger.warning(f"⚠️ 連續 {CIRCUIT_BREAK_THRESHOLD} 次抓到空資料，"
                                       f"研判 FinMind 限流，停止打 API，剩餘全用本地快取")
                else:
                    consecutive_empty = 0
                time.sleep(delay)
            else:
                cache_hits += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(tickers):
                logger.info(f"[{i+1}/{len(tickers)}] 基本面載入中... 快取:{cache_hits} API:{api_calls}"
                            + ("（API已斷路）" if api_disabled else ""))

        except Exception as e:
            logger.warning(f"[{i+1}/{len(tickers)}] {ticker} 基本面歷史載入失敗: {e}")
            cache[raw_ticker] = {"rev": pd.DataFrame(), "margin": pd.DataFrame(), "cashflow": pd.DataFrame(), "debt": pd.DataFrame()}

    status = "（中途斷路，部分用本地）" if api_disabled else ""
    logger.info(f"✅ 基本面序列載入完成：{cache_hits} 筆從快取，{api_calls} 筆打 API {status}")

    # ⚡ 效能優化：預先算好 available_date 並排序，避免回測迴圈裡每次 get_fundamental_at_date 重算
    for tk, d in cache.items():
        for key, avail_fn in (("rev", _revenue_available_date), ("margin", _margin_available_date), ("cashflow", _margin_available_date), ("debt", _margin_available_date)):
            dfx = d.get(key)
            if dfx is not None and not dfx.empty and "available_date" not in dfx.columns:
                try:
                    dfx = dfx.copy()
                    dfx["available_date"] = pd.to_datetime(dfx.index.map(avail_fn))
                    dfx = dfx.sort_values("available_date")
                    d[key] = dfx
                except Exception:
                    pass

    return cache
def fetch_monthly_revenue_yoy(ticker: str, months: int = 13, cache_only: bool = False) -> dict:
    """
    回傳單一標的最新月營收 YoY 摘要。
    資料來源優先序：
      1. 歷史序列 parquet（history/{ticker}.parquet）取最後幾筆 — 與回測共用同一份資料
      2. 若無歷史檔且 cache_only=False，才打 FinMind 補抓
    cache_only=True 時只讀本地，絕不打 API（掃描時用）。

    回傳格式：
    {
        "ticker": "2330",
        "latest_yoy": 35.2,
        "yoy_streak": 3,
        "consecutive_growth": True,
        "avg_yoy_3m": 28.5,
        "data_ok": True,
    }
    """
    result = {"ticker": ticker, "latest_yoy": None, "yoy_streak": 0,
              "consecutive_growth": False, "avg_yoy_3m": None,
              "mom": None, "mom_trend": "unknown", "data_ok": False}

    # ── 優先：從歷史序列 parquet 讀（與回測共用同一份資料）──
    hist_path = os.path.join(REVENUE_HISTORY_DIR, f"{ticker}.parquet")
    hist_df = None
    if os.path.exists(hist_path):
        try:
            hist_df = pd.read_parquet(hist_path)
        except Exception:
            hist_df = None

    # 歷史檔不存在或過期，且允許打 API → 補抓（會順便存成歷史 parquet）
    if (hist_df is None or hist_df.empty) and not cache_only:
        hist_df = fetch_revenue_history(ticker)

    if hist_df is None or hist_df.empty or "yoy" not in hist_df.columns:
        return result

    # 從歷史序列算摘要
    hist_df = hist_df.sort_index()
    yoy_series = hist_df["yoy"].dropna()
    # 基期異常保護：過濾掉 YoY 絕對值 > 500% 的雜訊（去年同月基期趨近 0 造成的假爆發）
    yoy_series = yoy_series[yoy_series.abs() <= 500]
    if yoy_series.empty:
        return result

    latest_yoy = float(yoy_series.iloc[-1])
    avg_yoy_3m = float(yoy_series.tail(3).mean().round(2))

    # MoM（月增率）：本月營收 vs 上月營收。從 revenue 序列直接算，零 API。
    mom = None
    mom_trend = "unknown"
    if "revenue" in hist_df.columns:
        rev_series = hist_df["revenue"].dropna()
        if len(rev_series) >= 2 and rev_series.iloc[-2] > 0:
            mom = round((rev_series.iloc[-1] - rev_series.iloc[-2]) / rev_series.iloc[-2] * 100, 2)
            # 近3個月 MoM 趨勢
            if len(rev_series) >= 4:
                recent_moms = rev_series.tail(4).pct_change().dropna() * 100
                if (recent_moms < 0).all():
                    mom_trend = "down"   # 連續下滑（轉折警訊）
                elif (recent_moms > 0).all():
                    mom_trend = "up"
                else:
                    mom_trend = "mixed"

    # 連續正成長月數（從最新往回數）
    streak = 0
    for y in reversed(yoy_series.tolist()):
        if y > 0:
            streak += 1
        else:
            break

    # 連續兩個月 YoY > 15%
    consecutive_growth = (
        len(yoy_series) >= 2
        and yoy_series.iloc[-1] >= 15
        and yoy_series.iloc[-2] >= 15
    )

    result.update({
        "latest_yoy":         round(latest_yoy, 2),
        "yoy_streak":         streak,
        "consecutive_growth": bool(consecutive_growth),
        "avg_yoy_3m":         avg_yoy_3m,
        "mom":                mom,
        "mom_trend":          mom_trend,
        "data_ok":            True,
    })
    return result


# ==========================================
# 毛利率質量篩選（季財報）
# ==========================================
def fetch_gross_margin(ticker: str, quarters: int = 6, cache_only: bool = False) -> dict:
    """
    抓取近 N 季毛利率，判斷是否在「擴張」狀態。
    cache_only=True 時只讀快取，不打 FinMind（掃描時用）。

    回傳格式：
    {
        "ticker": "2330",
        "latest_gm": 53.2,          # 最新季毛利率 (%)
        "gm_expanding": True,       # 最新季 > 前一季
        "gm_trend": "up",           # "up" / "down" / "flat"
        "data_ok": True,
    }
    """
    result = {"ticker": ticker, "latest_gm": None, "gm_expanding": False,
              "gm_trend": "unknown", "data_ok": False}

    # ── 優先：從毛利率歷史 parquet 讀（與回測共用同一份資料）──
    hist_path = os.path.join(MARGIN_HISTORY_DIR, f"{ticker}.parquet")
    hist_df = None
    if os.path.exists(hist_path):
        try:
            hist_df = pd.read_parquet(hist_path)
        except Exception:
            hist_df = None

    if hist_df is not None and not hist_df.empty and "gross_margin" in hist_df.columns:
        gm_series = hist_df["gross_margin"].dropna()
        if len(gm_series) >= 2:
            latest_gm = float(gm_series.iloc[-1])
            prev_gm   = float(gm_series.iloc[-2])
            import numpy as np
            vals = gm_series.tail(quarters).values
            slope = np.polyfit(range(len(vals)), vals, 1)[0] if len(vals) >= 2 else 0
            result.update({
                "latest_gm":    round(latest_gm, 2),
                "gm_expanding": latest_gm > prev_gm,
                "gm_trend":     "up" if slope > 0.3 else ("down" if slope < -0.3 else "flat"),
                "data_ok":      True,
            })
            return result

    # 歷史檔沒有或不足，且只讀快取模式 → 回傳空
    if cache_only:
        return {"ticker": ticker, "latest_gm": None, "gm_expanding": True,
                "gm_trend": "unknown", "data_ok": False}

    # 沒有歷史檔且允許打 API → 補抓（會存成歷史 parquet）
    hist_df = fetch_margin_history(ticker)
    if hist_df is not None and not hist_df.empty and "gross_margin" in hist_df.columns:
        gm_series = hist_df["gross_margin"].dropna()
        if len(gm_series) >= 2:
            latest_gm = float(gm_series.iloc[-1])
            prev_gm   = float(gm_series.iloc[-2])
            import numpy as np
            vals = gm_series.tail(quarters).values
            slope = np.polyfit(range(len(vals)), vals, 1)[0] if len(vals) >= 2 else 0
            result.update({
                "latest_gm":    round(latest_gm, 2),
                "gm_expanding": latest_gm > prev_gm,
                "gm_trend":     "up" if slope > 0.3 else ("down" if slope < -0.3 else "flat"),
                "data_ok":      True,
            })
    return result


def _fetch_gross_margin_legacy(ticker: str, quarters: int = 6, cache_only: bool = False) -> dict:
    """舊版：直接打 FinMind 財報 API（保留備用，主流程已改走歷史 parquet）"""
    cache = _load_cache(MARGIN_CACHE_FILE)
    if _cache_is_fresh(cache, ticker) and ticker in cache:
        return cache[ticker]

    if cache_only:
        return {"ticker": ticker, "latest_gm": None, "gm_expanding": True,
                "gm_trend": "unknown", "data_ok": False}

    start_date = (datetime.date.today() - datetime.timedelta(days=quarters * 100)).strftime("%Y-%m-%d")
    df = _finmind_get("TaiwanStockFinancialStatements", ticker, start_date)

    result = {"ticker": ticker, "latest_gm": None, "gm_expanding": False,
              "gm_trend": "unknown", "data_ok": False}

    if df.empty:
        _update_cache(cache, MARGIN_CACHE_FILE, ticker, result)
        return result

    # FinMind 財報欄位：type = 毛利率
    gm_df = df[df.get("type", pd.Series()).str.contains("毛利率|GrossMargin", case=False, na=False)].copy()
    if gm_df.empty:
        # 嘗試從 GrossProfit / Revenue 自己算
        gp = df[df.get("type", pd.Series()).str.contains("毛利$|GrossProfit", case=False, na=False)]
        rv = df[df.get("type", pd.Series()).str.contains("營業收入|Revenue", case=False, na=False)]
        if gp.empty or rv.empty:
            _update_cache(cache, MARGIN_CACHE_FILE, ticker, result)
            return result
        gp = gp[["date", "value"]].rename(columns={"value": "gp"})
        rv = rv[["date", "value"]].rename(columns={"value": "rv"})
        gm_df = gp.merge(rv, on="date")
        gm_df["gm"] = (pd.to_numeric(gm_df["gp"], errors="coerce") /
                       pd.to_numeric(gm_df["rv"], errors="coerce") * 100).round(2)
    else:
        gm_df["gm"] = pd.to_numeric(gm_df["value"], errors="coerce")

    gm_df = gm_df.dropna(subset=["gm"]).sort_values("date").tail(quarters)

    if len(gm_df) < 2:
        _update_cache(cache, MARGIN_CACHE_FILE, ticker, result)
        return result

    latest_gm   = float(gm_df["gm"].iloc[-1])
    prev_gm     = float(gm_df["gm"].iloc[-2])
    gm_expanding = latest_gm > prev_gm

    # 趨勢：用線性迴歸斜率判斷
    import numpy as np
    vals = gm_df["gm"].values
    slope = np.polyfit(range(len(vals)), vals, 1)[0]
    gm_trend = "up" if slope > 0.3 else ("down" if slope < -0.3 else "flat")

    result.update({
        "latest_gm":    latest_gm,
        "gm_expanding": gm_expanding,
        "gm_trend":     gm_trend,
        "data_ok":      True,
    })

    _update_cache(cache, MARGIN_CACHE_FILE, ticker, result)
    return result


# ==========================================
# 營業現金流（地雷股防禦）
# ==========================================
CASHFLOW_CACHE_FILE = os.path.join(CACHE_DIR, "cashflow_cache.json")

def fetch_operating_cashflow(ticker: str, quarters: int = 4, cache_only: bool = False) -> dict:
    """
    抓取最近的營業活動現金流，判斷是否為正。
    帳面賺錢但營業現金流為負，常是塞貨或做假帳的警訊。
    cache_only=True 時只讀快取，不打 FinMind。

    回傳：
    {
        "ticker": "2330",
        "latest_ocf": 12345.0,   # 最新一季營業現金流（千元）
        "ocf_positive": True,    # 是否為正
        "data_ok": True,
    }
    """
    cache = _load_cache(CASHFLOW_CACHE_FILE)
    if _cache_is_fresh(cache, ticker) and ticker in cache:
        return cache[ticker]

    result = {"ticker": ticker, "latest_ocf": None, "ocf_positive": False, "data_ok": False}

    if cache_only:
        return result

    start_date = (datetime.date.today() - datetime.timedelta(days=quarters * 100)).strftime("%Y-%m-%d")
    df = _finmind_get("TaiwanStockCashFlowsStatement", ticker, start_date)

    if df.empty:
        return result  # 抓失敗不寫快取，避免毒化

    ocf_df = df[df.get("type", pd.Series()).str.contains(
        "營業活動|CashFlowsFromOperating|OperatingActivities", case=False, na=False)].copy()

    if ocf_df.empty:
        return result

    ocf_df["ocf"] = pd.to_numeric(ocf_df["value"], errors="coerce")
    ocf_df = ocf_df.dropna(subset=["ocf"]).sort_values("date")

    if ocf_df.empty:
        return result

    latest_ocf = float(ocf_df["ocf"].iloc[-1])
    result.update({
        "latest_ocf":   latest_ocf,
        "ocf_positive": latest_ocf > 0,
        "data_ok":      True,
    })

    _update_cache(cache, CASHFLOW_CACHE_FILE, ticker, result)
    return result


# ==========================================
# 三大法人買賣超（籌碼面）
# ==========================================
INSTITUTIONAL_CACHE_FILE = os.path.join(CACHE_DIR, "institutional_cache.json")

def fetch_institutional_net(ticker: str, days: int = 10, cache_only: bool = False) -> dict:
    """
    抓取近 N 日三大法人買賣超，判斷外資是否連續買超。
    cache_only=True 時只讀快取（但籌碼面每天變，快取意義較小，掃描時建議當天現抓通過篩選的少數股）。

    回傳：
    {
        "ticker": "2330",
        "foreign_net": 12345,       # 最新一日外資買賣超（張，正=買超）
        "foreign_streak": 3,        # 外資連續買超天數
        "total_net_5d": 50000,      # 近5日三大法人合計買賣超（張）
        "data_ok": True,
    }
    """
    cache = _load_cache(INSTITUTIONAL_CACHE_FILE)
    # 籌碼面用「當天」快取（同一天內重複掃不重抓）
    if ticker in cache:
        c = cache[ticker]
        if c.get("_date") == datetime.date.today().isoformat():
            return c

    result = {"ticker": ticker, "foreign_net": None, "foreign_streak": 0,
              "total_net_5d": None, "data_ok": False}

    if cache_only:
        return result

    start_date = (datetime.date.today() - datetime.timedelta(days=days + 10)).strftime("%Y-%m-%d")
    df = _finmind_get("TaiwanStockInstitutionalInvestorsBuySell", ticker, start_date)

    if df.empty or "buy" not in df.columns or "sell" not in df.columns:
        return result  # 抓失敗不寫快取

    df["net"] = (pd.to_numeric(df["buy"], errors="coerce") -
                 pd.to_numeric(df["sell"], errors="coerce")) / 1000  # 股 → 張
    df["date"] = pd.to_datetime(df["date"])

    # 外資：FinMind 的 name 可能是 Foreign_Investor / Foreign_Dealer_Self 等
    foreign = df[df["name"].astype(str).str.contains("Foreign", case=False, na=False)].sort_values("date")
    if foreign.empty:
        return result

    foreign_daily = foreign.groupby("date")["net"].sum()
    foreign_net = float(foreign_daily.iloc[-1])

    # 外資連續買超天數
    streak = 0
    for n in reversed(foreign_daily.tolist()):
        if n > 0:
            streak += 1
        else:
            break

    # 近5日三大法人合計
    total_5d = float(df[df["date"] >= df["date"].max() - pd.Timedelta(days=7)]["net"].sum())

    result.update({
        "foreign_net":    round(foreign_net, 1),
        "foreign_streak": streak,
        "total_net_5d":   round(total_5d, 1),
        "data_ok":        True,
        "_date":          datetime.date.today().isoformat(),
    })

    _update_cache(cache, INSTITUTIONAL_CACHE_FILE, ticker, result)
    return result


# ==========================================
# 批量篩選入口（板塊層級）
# ==========================================
def screen_sector_by_revenue(
    sector_tickers: list,
    min_yoy: float = 15.0,
    require_consecutive: bool = True,
    require_gm_expanding: bool = False,
    require_positive_cashflow: bool = False,
    delay: float = 0.3,
    cache_only: bool = False,
) -> pd.DataFrame:
    """
    對一個板塊的所有標的做月營收篩選，回傳通過條件的 DataFrame。
    cache_only=True 時只讀快取，不打 FinMind（掃描時用，避免限流）。

    columns: ticker, latest_yoy, avg_yoy_3m, mom, yoy_streak, consecutive_growth,
             latest_gm, gm_expanding, gm_trend, ocf_positive, pass_filter
    """
    rows = []
    for raw_ticker in sector_tickers:
        ticker = str(raw_ticker).replace(".TW", "")
        rev = fetch_monthly_revenue_yoy(ticker, cache_only=cache_only)
        # 毛利率一律讀取顯示（讀本地 parquet 不花 API）；只有勾選時才當篩選條件
        gm  = fetch_gross_margin(ticker, cache_only=cache_only)
        # 營業現金流：一律讀取顯示，只有勾選時才當篩選條件
        ocf = fetch_operating_cashflow(ticker, cache_only=cache_only)

        pass_revenue = (
            rev["data_ok"]
            and rev["latest_yoy"] is not None
            and rev["latest_yoy"] >= min_yoy
            and (not require_consecutive or rev["consecutive_growth"])
        )
        pass_margin = not require_gm_expanding or gm["gm_expanding"]
        # 現金流過濾：勾選時，抓得到資料且為負才擋；抓不到不誤殺
        pass_cashflow = (
            not require_positive_cashflow
            or (not ocf["data_ok"])           # 抓不到資料不擋
            or ocf["ocf_positive"]
        )
        pass_filter = pass_revenue and pass_margin and pass_cashflow

        rows.append({
            "ticker":             f"{ticker}.TW",
            "latest_yoy":         rev["latest_yoy"],
            "avg_yoy_3m":         rev["avg_yoy_3m"],
            "mom":                rev.get("mom"),
            "mom_trend":          rev.get("mom_trend", "unknown"),
            "yoy_streak":         rev["yoy_streak"],
            "consecutive_growth": rev["consecutive_growth"],
            "latest_gm":          gm.get("latest_gm"),
            "gm_expanding":       gm.get("gm_expanding"),
            "gm_trend":           gm.get("gm_trend"),
            "ocf_positive":       ocf.get("ocf_positive"),
            "pass_filter":        pass_filter,
        })
        if not cache_only:
            time.sleep(delay)  # 避免打爆 FinMind rate limit

    return pd.DataFrame(rows)


def get_sector_revenue_score(sector_tickers: list, min_yoy: float = 10.0, cache_only: bool = False) -> float:
    """
    計算一個板塊的「月營收動能分數」（供 SectorRadar 排名用）。
    分數 = 板塊內通過 min_yoy 的股票比例 × 平均 YoY
    cache_only=True 時只讀快取，不打 FinMind（掃描時用）。
    """
    if not sector_tickers:
        return 0.0

    yoys = []
    for raw_ticker in sector_tickers:
        ticker = str(raw_ticker).replace(".TW", "")
        _cache = _load_cache(REVENUE_CACHE_FILE)
        _is_cached = _cache_is_fresh(_cache, ticker) and ticker in _cache
        rev = fetch_monthly_revenue_yoy(ticker, cache_only=cache_only)
        if rev["data_ok"] and rev["latest_yoy"] is not None:
            yoys.append(rev["latest_yoy"])
        if not _is_cached and not cache_only:
            time.sleep(0.2)

    if not yoys:
        return 0.0

    pass_rate = sum(1 for y in yoys if y >= min_yoy) / len(yoys)
    avg_yoy   = sum(yoys) / len(yoys)
    return round(pass_rate * avg_yoy, 2)


def rank_sectors_at_date(sector_map: dict, query_date, fundamental_cache: dict,
                          top_n: int = 3, min_yoy: float = 10.0,
                          price_cache: dict = None, yoy_cap: float = 100.0,
                          price_weight: float = 0.4) -> list:
    """
    Point-in-time 板塊輪動：用「query_date 當下已公告」的月營收，
    算各板塊動能分數，回傳最強的 top_n 個板塊名稱。
    無前視偏差（透過 get_fundamental_at_date 的 available_date 機制）。

    評分邏輯（v2，修正景氣循環股灌水問題）：
      1. YoY 上限封頂（yoy_cap，預設 100%）：避免景氣循環股/低基期股的爆衝 YoY 灌水
      2. 用中位數 YoY 而非平均：少數爆衝股無法拉高整個板塊分數
      3. 加股價動能權重：真正有資金流入的板塊，股價會領先反映
         price_cache 有提供時，板塊分數 = (1-price_weight)×營收分 + price_weight×股價動能分
         price_cache 沒提供時，退回純營收分（向後相容）

    sector_map: {板塊名: [ticker, ...]}
    fundamental_cache: {ticker: {"rev": df, "margin": df, ...}} 預載好的歷史序列
    price_cache: {ticker: df} 完整 K 線（含 Close），算 60 日股價動能用
    回傳: [最強板塊名, 第二強, ...] 共 top_n 個
    """
    import numpy as np

    query_ts = pd.Timestamp(query_date)

    # ── Step 1：算各板塊的月營收分數（中位數 YoY + 封頂）──
    raw_rev = {}
    for sector, tickers in sector_map.items():
        yoys = []
        for raw_t in tickers:
            t = str(raw_t).replace(".TW", "")
            fc = fundamental_cache.get(f"{t}.TW") or fundamental_cache.get(t)
            if not fc:
                continue
            rev_hist = fc.get("rev")
            if rev_hist is None or rev_hist.empty:
                continue
            fund = get_fundamental_at_date(t, query_date, rev_history=rev_hist)
            if fund["latest_yoy"] is not None:
                # ✅ 修正1：YoY 封頂，景氣循環股 +300% 一律當 +100% 算
                capped = min(fund["latest_yoy"], yoy_cap)
                yoys.append(capped)
        if yoys:
            pass_rate  = sum(1 for y in yoys if y >= min_yoy) / len(yoys)
            # ✅ 修正2：用中位數而非平均，少數爆衝股無法灌水
            median_yoy = float(np.median(yoys))
            raw_rev[sector] = pass_rate * median_yoy

    # ── Step 2：算各板塊的股價動能分數（60 日漲幅中位數）──
    raw_price = {}
    if price_cache:
        for sector, tickers in sector_map.items():
            moms = []
            for raw_t in tickers:
                df = price_cache.get(raw_t) or price_cache.get(f"{str(raw_t).replace('.TW','')}.TW")
                if df is None or df.empty or "Close" not in df.columns:
                    continue
                past = df[df.index <= query_ts]
                if len(past) >= 60:
                    mom = (past["Close"].iloc[-1] / past["Close"].iloc[-60]) - 1
                    moms.append(mom)
            if moms:
                # 中位數股價動能，同樣避免單一飆股拉高板塊
                raw_price[sector] = float(np.median(moms))

    # ── Step 3：標準化後合併（若無股價資料則退回純營收）──
    if not raw_rev and not raw_price:
        return []

    def _normalize(d):
        if not d:
            return {}
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    # 沒有股價資料 → 純營收排名（向後相容）
    if not raw_price:
        ranked = sorted(raw_rev.items(), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:top_n]]

    norm_rev   = _normalize(raw_rev)
    norm_price = _normalize(raw_price)

    composite = {}
    for sector in set(norm_rev) | set(norm_price):
        r = norm_rev.get(sector, 0.0)
        p = norm_price.get(sector, 0.0)
        # 兩個訊號都有才完整計分；只有一個的打折（避免單一訊號板塊插隊）
        if sector in norm_rev and sector in norm_price:
            composite[sector] = (1 - price_weight) * r + price_weight * p
        elif sector in norm_rev:
            composite[sector] = r * 0.6
        else:
            composite[sector] = p * 0.4

    ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]


# ==========================================
# 快取更新輔助
# ==========================================
def check_fundamental_exit(
    ticker: str,
    entry_date,
    current_date,
) -> dict:
    """
    判斷持倉中的標的是否觸發基本面出場條件。
    任一條件成立即建議出場。

    條件一：月營收 YoY 轉負（最新月 YoY < 0）
    條件二：最新季毛利率 < 前一季（毛利率衰退）

    回傳格式：
    {
        "should_exit": False,
        "reason": "",           # "YoY轉負" / "毛利衰退" / "YoY轉負+毛利衰退" / ""
        "latest_yoy": 12.3,
        "gm_expanding": True,
    }
    """
    result = {
        "should_exit": False,
        "reason": "",
        "latest_yoy": None,
        "gm_expanding": None,
    }

    reasons = []

    # 條件一：月營收 YoY 轉負
    rev = fetch_monthly_revenue_yoy(str(ticker).replace(".TW", ""))
    if rev["data_ok"] and rev["latest_yoy"] is not None:
        result["latest_yoy"] = rev["latest_yoy"]
        if rev["latest_yoy"] < 0:
            reasons.append("YoY轉負")

    # 條件二：毛利率衰退
    gm = fetch_gross_margin(str(ticker).replace(".TW", ""))
    if gm["data_ok"]:
        result["gm_expanding"] = gm["gm_expanding"]
        if not gm["gm_expanding"]:
            reasons.append("毛利衰退")

    if reasons:
        result["should_exit"] = True
        result["reason"]      = "+".join(reasons)

    return result


def _update_cache(cache: dict, path: str, key: str, value: dict):
    cache[key] = value
    if "__meta__" not in cache:
        cache["__meta__"] = {}
    cache["__meta__"][key] = {"date": datetime.date.today().isoformat()}
    _save_cache(path, cache)