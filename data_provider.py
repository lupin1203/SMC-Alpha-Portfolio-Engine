import yfinance as yf
import pandas as pd
import requests
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

def fetch_stock_data(ticker_str, days):
    stock_id = str(ticker_str).split(".")[0]
    local_path = f"./local_data/{stock_id}.parquet"
    
    # 🟢 第一防線：從本地 Data Lake 載入
    if os.path.exists(local_path):
        try:
            df = pd.read_parquet(local_path)
            # 確保索引是 datetime，且欄位名稱首字母大寫
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
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
        stock = yf.Ticker(f"{stock_id}.TW")
        df_result = stock.history(period="max", auto_adjust=True)
        if not df_result.empty:
            df_result = df_result[df_result.index >= pd.to_datetime(start_date)]
            df_result = df_result[['Open', 'High', 'Low', 'Close', 'Volume']]
            df_result.index = df_result.index.tz_localize(None)
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