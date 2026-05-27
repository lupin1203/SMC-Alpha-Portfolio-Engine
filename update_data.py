# update_data.py
import yfinance as yf
import pandas as pd
import os
import time
import random
import logging
from data_provider import load_data # 借用你原本寫好的函數來取得所有股票清單

# 初始化設定
DATA_DIR = "./local_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def download_all_stocks(years=10):
    # 取得你系統中所有的股票代碼
    INDUSTRY_DICT, _, _ = load_data()
    all_tickers = []
    for tickers in INDUSTRY_DICT.values():
        all_tickers.extend(tickers)
    
    # 確保代碼不重複
    all_tickers = list(set(all_tickers))
    total = len(all_tickers)
    success_count, fail_count = 0, 0

    logging.info(f"啟動全市場資料落地計畫，共計 {total} 檔標的，預計耗時約 30-60 分鐘...")

    for i, ticker in enumerate(all_tickers):
        file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
        
        logging.info(f"[{i+1}/{total}] 正在同步 {ticker} ...")
        try:
            # 這裡以 yfinance 為例，加上 .TW 後綴
            # 如果你原本是用 FinMind，可以替換成 FinMind 的 API
            df = yf.download(f"{ticker}.TW", period=f"{years}y", progress=False)
            
            if df is not None and not df.empty and len(df) > 10:
                # 存成 Parquet 格式
                df.to_parquet(file_path)
                success_count += 1
            else:
                logging.warning(f"[{i+1}/{total}] {ticker} 無有效資料。")
                fail_count += 1
                
        except Exception as e:
            logging.error(f"[{i+1}/{total}] 下載 {ticker} 失敗: {e}")
            fail_count += 1

        # 🌟 終極防禦：隨機延遲 0.5 到 2 秒，裝成人類慢慢查資料，絕對不會被 Ban IP
        time.sleep(random.uniform(0.5, 2.0))

    logging.info(f"🎉 任務結束！成功落地: {success_count} 檔，失敗/無資料: {fail_count} 檔。")

if __name__ == "__main__":
    download_all_stocks(years=10)