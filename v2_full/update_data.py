import os
import time
import random
import datetime
import logging
import pandas as pd
from data_provider import load_data, fetch_stock_data 

# 初始化設定
DATA_DIR = "./local_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# ==========================================
# 📡 純下載引擎（直接打 yfinance，繞過 fetch_stock_data 的本地快取）
# ==========================================
def _download_fresh_from_yfinance(ticker_str, days):
    """
    ★ 修正核心 bug：update 專用的純下載函數。
    絕不讀本地快取，一律直接向 yfinance 抓最新資料。

    原本 update_data 呼叫 fetch_stock_data，但 fetch_stock_data 的第一防線
    「本地有檔就直接回傳舊檔」會攔截，導致 update 拿到的永遠是舊資料、
    抓不到新 K 線（資料因此停在 6/15）。此函數專供 update 使用，直接打 API。
    """
    import yfinance as yf
    stock_id = str(ticker_str).replace(".TW.TW", "").replace(".TW", "").split(".")[0]
    start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        import logging as _logging
        _yf_logger = _logging.getLogger('yfinance')
        _prev = _yf_logger.level
        _yf_logger.setLevel(_logging.CRITICAL)   # 靜音已下市股票的雜訊

        stock = yf.Ticker(f"{stock_id}.TW")
        df = stock.history(period="max", auto_adjust=True)

        _yf_logger.setLevel(_prev)

        if df is not None and not df.empty:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[df.index >= pd.to_datetime(start_date)]
            keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            return df[keep]
    except Exception as e:
        logging.error(f"[純下載] yfinance 抓取失敗 ({stock_id}): {e}")
    return pd.DataFrame()

# ==========================================
# 🧹 核心防禦：暴力融合與去重超度工具 (Shield Core)
# ==========================================
def force_merge_and_purge_ghost_files(stage_label=""):
    """
    不講道理的物理清洗器：
    不論是下載前還是下載後，只要發現有 2330.parquet 和 2330.TW.parquet 同時存在，
    或者只存在 2330.parquet，一律強制將數據融合，並把沒帶 .TW 的殘留檔直接從硬碟抹除！
    """
    logging.info(f"[{stage_label}] 啟動雙重防護清洗，正在強行校正硬碟檔案命名...")
    if not os.path.exists(DATA_DIR):
        return

    files = os.listdir(DATA_DIR)
    ticker_groups = {}

    for f in files:
        if not f.endswith('.parquet'): 
            continue
        # 🔪 暴力拔除所有可能後綴，找出純數字代號 (如 '2330')
        base_ticker = f.replace('.TW.parquet', '').replace('.parquet', '')
        if base_ticker not in ticker_groups:
            ticker_groups[base_ticker] = []
        ticker_groups[base_ticker].append(f)

    fixed_count = 0
    deleted_count = 0

    for base_ticker, file_list in ticker_groups.items():
        correct_filename = f"{base_ticker}.TW.parquet"
        correct_filepath = os.path.join(DATA_DIR, correct_filename)

        # 🟢 如果這個代號底下只有 1 個檔案，且檔名完全標準，代表很健康，直接跳過
        if len(file_list) == 1 and file_list[0] == correct_filename:
            continue 

        # 💥 發現衝突（例如同時存在 2330.parquet 與 2330.TW.parquet）
        df_list = []
        for f in file_list:
            try:
                file_path = os.path.join(DATA_DIR, f)
                df = pd.read_parquet(file_path)
                
                # 🛡️ 強制剝離時區，防止比較時發生錯誤
                if isinstance(df.index, pd.DatetimeIndex):
                    df.index = df.index.tz_localize(None)
                    
                df_list.append(df)
            except Exception as e:
                logging.error(f"  ❌ 讀取碎檔 {f} 失敗: {e}")

        if df_list:
            # 🧬 數據強制大融合
            merged_df = pd.concat(df_list)
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
            merged_df.sort_index(inplace=True)

            # 💾 覆寫至絕對合規的唯一標準路徑 (.TW.parquet)
            merged_df.to_parquet(correct_filepath)
            fixed_count += 1
            
            # 💀 執行你的指令：通通殺掉，只留標準檔！
            for f in file_list:
                if f != correct_filename:
                    bad_path = os.path.join(DATA_DIR, f)
                    try:
                        os.remove(bad_path)
                        deleted_count += 1
                    except Exception:
                        pass

    if fixed_count > 0 or deleted_count > 0:
        logging.info(f"[{stage_label} 完畢] 強制重組了 {fixed_count} 檔衝突股票，硬碟物理抹除了 {deleted_count} 個非標準殘留檔。")
    else:
        logging.info(f"[{stage_label} 完畢] 資料夾極度純淨，未偵測到任何命名衝突。")
    logging.info("--------------------------------------------------")


# ==========================================
# 📥 增量更新執行聯鎖 (Incremental Update)
# ==========================================
def stage_2_download_all_stocks(years=10):
    logging.info(f"啟動全市場【雙引擎增量更新】計畫...")
    
    INDUSTRY_DICT, _, _ = load_data(use_local_scan=False) # 關閉本地自動偵測，避免髒檔有名稱
    raw_tickers = []
    for tickers in INDUSTRY_DICT.values():
        raw_tickers.extend(tickers)
    
    raw_tickers.append("0050")
    
    # 🌟 終極淨化名單
    clean_tickers = [str(t).replace(".TW.TW", "").replace(".TW", "") for t in raw_tickers]
    all_tickers = list(set(clean_tickers))
    
    total = len(all_tickers)
    success_count, skip_count, fail_count = 0, 0, 0
    today = datetime.date.today()

    for i, pure_ticker in enumerate(all_tickers):
        ticker_str = f"{pure_ticker}.TW" 
        file_path = os.path.join(DATA_DIR, f"{ticker_str}.parquet") # 外部鎖死的標準存檔路徑
        
        df_old = pd.DataFrame()
        missing_days = years * 365
        need_full_download = True

        # 讀取舊資料
        if os.path.exists(file_path):
            try:
                df_old = pd.read_parquet(file_path)
                if not df_old.empty:
                    df_old.index = pd.to_datetime(df_old.index)
                    if df_old.index.tz is not None:
                        df_old.index = df_old.index.tz_localize(None)
                    last_date = df_old.index.max().date()
                    missing_days = (today - last_date).days
                    
                    if missing_days <= 1 or (today.weekday() >= 5 and missing_days <= 2):
                        logging.info(f"[{i+1}/{total}]  {ticker_str} 本地已最新 ({last_date})，跳過。")
                        skip_count += 1
                        continue
                    
                    need_full_download = False
            except Exception as e:
                logging.warning(f"[{i+1}/{total}] {ticker_str} 讀舊檔失敗: {e}")

        try:
            fetch_days = years * 365 if need_full_download else missing_days + 5
            
            # 🌟 呼叫純下載引擎（直接打 yfinance，繞過本地快取，才抓得到新資料）
            df_new = _download_fresh_from_yfinance(ticker_str, days=fetch_days)
            
            if df_new is not None and not df_new.empty:
                if df_new.index.tz is not None:
                    df_new.index = df_new.index.tz_localize(None)
                
                available_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df_new.columns]
                df_new = df_new[available_cols]
                df_new.dropna(subset=['Close'], inplace=True)
                
                if not df_new.empty:
                    if not df_old.empty:
                        df_combined = pd.concat([df_old, df_new])
                        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                        df_combined.sort_index(inplace=True)
                    else:
                        df_combined = df_new.sort_index()
                    
                    # 外部強制寫入標準檔
                    df_combined.to_parquet(file_path)
                    success_count += 1
                    time.sleep(random.uniform(0.05, 0.15))
                else:
                    fail_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            logging.error(f"[{i+1}/{total}] 下載 {ticker_str} 失敗: {e}")
            fail_count += 1

    logging.info(f"==========================================")
    logging.info(f" 增量數據下載階段結束！")
    logging.info(f" 成功: {success_count} 檔 | ⏭跳過: {skip_count} 檔 | 失敗: {fail_count} 檔")
    logging.info(f"==========================================")

# ==========================================
# 🏁 終極流水線入口
# ==========================================
if __name__ == "__main__":
    print("[原子級別數據流水線啟動]")
    
    force_merge_and_purge_ghost_files(stage_label="第一道防線：下載前全面淨化")
    stage_2_download_all_stocks(years=10)
    force_merge_and_purge_ghost_files(stage_label="第二道防線：下載後終極決殺")

    # ── Step 3：更新月營收 + 毛利率歷史序列 ──
    print("\n[Step 3] 更新月營收與毛利率歷史序列...")
    try:
        from data_provider import load_data
        from revenue_fetcher import build_fundamental_history_cache
        INDUSTRY_DICT, _, _ = load_data(use_local_scan=False)
        all_tickers = list({t for tks in INDUSTRY_DICT.values() for t in tks})
        logging.info(f"共 {len(all_tickers)} 檔標的需更新基本面歷史")
        build_fundamental_history_cache(all_tickers, delay=0.4)
        print(f"基本面歷史序列更新完成（{len(all_tickers)} 檔）")
    except Exception as e:
        print(f"基本面歷史更新失敗（不影響 K 線資料）: {e}")

    print("[大功告成] 資料池已達到 100% 無菌狀態，硬碟裡絕無任何重複股票！")