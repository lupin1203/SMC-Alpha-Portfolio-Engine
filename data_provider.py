import yfinance as yf
import pandas as pd
import datetime
import logging
import time
import random
import os 
import glob
from config import DEFAULT_SEARCH_LIST, DEFAULT_STOCK_NAMES, DEFAULT_INDUSTRY_DICT

def load_data(use_local_scan=True):
    """
    V19.0 智能載入：優先讀取 CSV，並自動偵測 local_data 擴充標的。
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
            local_tickers = [os.path.basename(f).replace(".parquet", "") for f in all_files]
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

def split_ticker(ticker_str):
    """'2330.TW' → ('2330', 'TW')；'6488.TWO' → ('6488', 'TWO')；沒有後綴時預設上市 (.TW)。"""
    parts = str(ticker_str).strip().split(".")
    suffix = parts[1].upper() if len(parts) > 1 and parts[1] else "TW"
    return parts[0], suffix


def latest_local_date(ticker_str="0050.TW"):
    """本機快取裡這檔股票最後一根 K 線的日期；沒有快取回傳 None。用來檢查資料是不是太舊。"""
    stock_id, _ = split_ticker(ticker_str)
    local_path = f"./local_data/{stock_id}.parquet"
    if not os.path.exists(local_path):
        return None
    try:
        idx = pd.to_datetime(pd.read_parquet(local_path, columns=[]).index)
        return idx.max().tz_localize(None) if idx.tz is not None else idx.max()
    except Exception as e:
        logging.warning(f"讀取本地緩存日期失敗 {stock_id}: {e}")
        return None


def fetch_stock_data(ticker_str, days):
    """
    讀取 K 線。本機有快取 (./local_data/<代號>.parquet) 就直接用，**不會自動更新**；
    要最新資料請先執行 python update_data.py。沒有快取才從 yfinance 下載最近 days 天。
    """
    stock_id, suffix = split_ticker(ticker_str)
    local_path = f"./local_data/{stock_id}.parquet"
    
    # 🟢 第一防線：從本地 Data Lake 載入
    if os.path.exists(local_path):
        try:
            df = pd.read_parquet(local_path)
            # 確保索引是 datetime，且欄位名稱首字母大寫
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # 舊版快取可能帶時區 (Asia/Taipei)，統一轉成不帶時區，避免與一般日期比較時報錯
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
    
    # 優先嘗試 yfinance (更為穩定)
    try:
        stock = yf.Ticker(f"{stock_id}.{suffix}")
        df_result = stock.history(period="max", auto_adjust=True)
        if not df_result.empty:
            # 先去掉時區再比較日期 (帶時區的索引不能和一般日期直接比較)
            df_result.index = df_result.index.tz_localize(None)
            df_result = df_result[df_result.index >= pd.to_datetime(start_date)]
            df_result = df_result[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logging.error(f"YFinance 下載失敗 ({stock_id}): {e}")
            
    # 第三步：存入本地快取
    if not df_result.empty:
        try:
            if not os.path.exists("./local_data"):
                os.makedirs("./local_data")
            df_result.to_parquet(local_path)
        except Exception as e:
            logging.warning(f"寫入本地緩存失敗 {stock_id}: {e}")
            
    return df_result