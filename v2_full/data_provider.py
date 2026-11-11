import yfinance as yf
import pandas as pd
import datetime
import logging
import time
import random
import os
import glob
from config import DEFAULT_SEARCH_LIST, DEFAULT_STOCK_NAMES, DEFAULT_INDUSTRY_DICT

DATA_DIR = "./local_data"  # 供外部模組引用

def load_data(use_local_scan=True):
    """
    V20.0 智能載入：優先讀取 CSV，並自動偵測 local_data 擴充標的。
    """
    try:
        if not os.path.exists("stock_list.csv"):
            return DEFAULT_INDUSTRY_DICT, DEFAULT_STOCK_NAMES, DEFAULT_SEARCH_LIST
            
        df = pd.read_csv("stock_list.csv")
        industry_map = df.groupby('category')['ticker'].apply(list).to_dict()
        name_map = pd.Series(df.name.values, index=df.ticker).to_dict()
        search_list = (df['ticker'] + " - " + df['name']).tolist()
        
        if use_local_scan and os.path.exists("./local_data"):
            all_files = glob.glob("./local_data/*.parquet")
            # 這裡的邏輯也修正，直接取最乾淨的代號來比對
            local_tickers = [os.path.basename(f).replace(".TW.parquet", "").replace(".parquet", "") for f in all_files]
            existing_tickers = [str(t).replace('.TW', '') for t in df['ticker'].tolist()]
            
            missing_tickers = [t for t in local_tickers if t not in existing_tickers]
            
            if missing_tickers:
                industry_map["自動偵測(全市場)"] = [f"{t}.TW" for t in missing_tickers]
                for t in missing_tickers:
                    name_map[f"{t}.TW"] = "未知名稱"
                    search_list.append(f"{t}.TW - 未知名稱")
                    
        return industry_map, name_map, search_list
    except Exception as e:
        logging.error(f"資料載入失敗: {e}")
        return DEFAULT_INDUSTRY_DICT, DEFAULT_STOCK_NAMES, DEFAULT_SEARCH_LIST

def fetch_index_data(index_symbol, days=250):
    """
    抓取指數資料（如 ^TWII 加權、^TWOII 櫃買、0050.TW）。
    指數代號不剝 .TW、不加 .TW，直接傳給 yfinance。
    0050.TW 仍走原本的個股快取邏輯。
    """
    # 0050 走原本的個股邏輯（有本地快取）
    if index_symbol == "0050.TW":
        return fetch_stock_data("0050.TW", days=days)

    import datetime as _dt
    start_date = (_dt.date.today() - _dt.timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        import logging as _logging
        _yf_logger = _logging.getLogger('yfinance')
        _prev = _yf_logger.level
        _yf_logger.setLevel(_logging.CRITICAL)

        idx = yf.Ticker(index_symbol)
        df = idx.history(period="max", auto_adjust=True)

        _yf_logger.setLevel(_prev)

        if df is not None and not df.empty:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[df.index >= pd.to_datetime(start_date)]
            keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            return df[keep]
    except Exception as e:
        logging.error(f"指數下載失敗 ({index_symbol}): {e}")
    return pd.DataFrame()


def fetch_stock_data(ticker_str, days):
    """抓取股票資料，確保檔名 100% 統一為 .TW.parquet"""
    # 確保不管傳進來的是 2330 還是 2330.TW，都剝乾淨只留 2330
    stock_id = str(ticker_str).replace(".TW.TW", "").replace(".TW", "").split(".")[0]
    
    # 💥 強制鎖死：檔名絕對要有 .TW
    local_path = f"./local_data/{stock_id}.TW.parquet"
    
    # 🟢 第一防線：從本地 Data Lake 載入
    if os.path.exists(local_path):
        try:
            df = pd.read_parquet(local_path)
            # 確保索引是 datetime，且欄位名稱首字母大寫
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            # 安全剝除時區
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            df.columns = [c.capitalize() if c.lower() != 'fvg' else 'FVG' for c in df.columns]
            return df
        except Exception as e:
            logging.warning(f"讀取本地緩存失敗 {stock_id}: {e}")

    # 🔴 第二防線：API 抓取
    start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    time.sleep(random.uniform(0.1, 0.4))

    df_result = pd.DataFrame()

    # 優先嘗試 yfinance，靜音已下市股票的錯誤訊息
    try:
        import logging as _logging
        _yf_logger = _logging.getLogger('yfinance')
        _prev_level = _yf_logger.level
        _yf_logger.setLevel(_logging.CRITICAL)  # 只保留最嚴重的錯誤

        stock = yf.Ticker(f"{stock_id}.TW")
        df_result = stock.history(period="max", auto_adjust=True)

        _yf_logger.setLevel(_prev_level)  # 恢復原本的 log level

        if not df_result.empty:
            # ✅ 先剝除時區，再做日期比較（避免 tz-aware vs tz-naive 比較錯誤）
            if df_result.index.tz is not None:
                df_result.index = df_result.index.tz_localize(None)

            df_result = df_result[df_result.index >= pd.to_datetime(start_date)]
            df_result = df_result[['Open', 'High', 'Low', 'Close', 'Volume']]
                
    except Exception as e:
        logging.error(f"YFinance 下載失敗 ({stock_id}): {e}")
            
    # 第三步：存入本地快取 (存檔路徑已經強制鎖定為 .TW.parquet)
    if not df_result.empty:
        try:
            if not os.path.exists("./local_data"):
                os.makedirs("./local_data")
            df_result.to_parquet(local_path)
        except Exception as e:
            logging.warning(f"寫入本地緩存失敗 {stock_id}: {e}")
            
    return df_result