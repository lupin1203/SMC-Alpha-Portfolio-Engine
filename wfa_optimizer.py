import pandas as pd
import numpy as np
from datetime import timedelta
import logging

# 引入你的策略與回測核心
from data_provider import fetch_stock_data
from strategy import apply_smc_logic
from backtest_engine import run_backtest_core

# === WFA 參數設定 ===
TRAIN_DAYS = 500  # 訓練期 (約 2 年)
TEST_DAYS = 250   # 盲測期 (約 1 年)
TICKER = "2376.TW" # 測試指標股 (對應你的檔名，例如 2376.TW.csv)

# 🌟 記得把這裡換成你電腦裡實際放 CSV 的資料夾路徑！
# 注意：Windows 路徑建議用雙反斜線 \\ 或是單斜線 /
DATA_FOLDER = "C:/Users/tame1/OneDrive/桌面/SMC-/data" 

# 我們要讓系統自己尋找的「參數池」
PARAM_GRID = {
    'gap_pct': [0.2, 0.3, 0.4, 0.5, 0.6]
}

# 🌟 這是你剛才漏掉的最重要的一行！(函數宣告)
def run_wfa_for_stock(ticker):
    print(f"啟動 {ticker} 的 Walk Forward Analysis...")
    
    # --- 讀取本地資料 ---
    df_raw = fetch_stock_data(ticker, days=2500) # 抓過去約 7-10 年
    if df_raw is None or df_raw.empty:
        print(f"🚨 無法從網路抓取 {ticker} 的歷史資料，自動跳過。")
        return
        
    # 確保資料格式正確
    df_raw['Date'] = pd.to_datetime(df_raw.index)
    # 確保資料是按照時間排序的
    df_raw.sort_index(inplace=True)
    df_raw['Date'] = pd.to_datetime(df_raw.index)
    
    start_date = df_raw['Date'].min()
    end_date = df_raw['Date'].max()
    
    current_train_start = start_date
    oos_trades = [] # 儲存所有盲測期的真實交易
    
    # --- 🔄 開始滾動時間窗格 ---
    while True:
        train_end = current_train_start + timedelta(days=TRAIN_DAYS)
        test_end = train_end + timedelta(days=TEST_DAYS)
        
        if test_end > end_date:
            break # 數據用完了，結束滾動
            
        print(f"\n🗓️ 訓練期: {current_train_start.date()} ~ {train_end.date()}")
        print(f"🎯 盲測期: {train_end.date()} ~ {test_end.date()}")
        
        # 1. 切割訓練資料
        train_df = df_raw[(df_raw['Date'] >= current_train_start) & (df_raw['Date'] < train_end)].copy()
        
        best_param = None
        best_profit = -9999
        
        # 2. 煉丹：在訓練期尋找最佳參數 (In-Sample Optimization)
        for gap in PARAM_GRID['gap_pct']:
            df_logic = apply_smc_logic(train_df.copy(), min_gap_pct=gap, use_sma_filter=True)
            res_df, *_ = run_backtest_core(df_logic, bt_mode='fixed_rr', bt_val=3.0, slippage_pct=0.1, fee_disc=0.5, pen_pct=0.1, min_vol=1000)
            
            profit = res_df['真實R'].sum() if (res_df is not None and not res_df.empty) else 0
            if profit > best_profit:
                best_profit = profit
                best_param = gap
                
        print(f"🏆 訓練期最佳參數: 缺口 {best_param}% (獲利 {round(best_profit,2)}R)")
        
        # 3. 盲測：用找出的最佳參數，去跑未來的測試期 (Out-of-Sample)
        test_df = df_raw[(df_raw['Date'] >= train_end) & (df_raw['Date'] < test_end)].copy()
        
        # 確保有足夠的資料算均線等前期指標 (往前抓 100 天緩衝)
        history_buffer = df_raw[(df_raw['Date'] >= train_end - timedelta(days=100)) & (df_raw['Date'] < train_end)].copy()
        test_df_with_history = pd.concat([history_buffer, test_df])
        
        df_test_logic = apply_smc_logic(test_df_with_history, min_gap_pct=best_param, use_sma_filter=True)
        # 只取盲測期內的訊號
        df_test_logic = df_test_logic[df_test_logic['Date'] >= train_end] 
        
        res_test, *_ = run_backtest_core(df_test_logic, bt_mode='fixed_rr', bt_val=3.0, slippage_pct=0.1, fee_disc=0.5,pen_pct=0.1, min_vol=1000 )
        
        if res_test is not None and not res_test.empty:
            res_test['盲測年份'] = train_end.year
            res_test['使用參數'] = best_param
            oos_trades.append(res_test)
            print(f"📈 盲測期表現: {round(res_test['真實R'].sum(),2)}R")
        else:
            print("📉 盲測期表現: 0R (無交易)")
            
        # 推進窗格：前進一個測試期
        current_train_start += timedelta(days=TEST_DAYS)

    # === 統計所有盲測結果 ===
    if oos_trades:
        final_oos_df = pd.concat(oos_trades)
        total_oos_r = final_oos_df['真實R'].sum()
        print("\n" + "="*40)
        print("🔥 Walk Forward 盲測最終報告 🔥")
        print("="*40)
        print(f"總盲測淨利: {round(total_oos_r, 2)} R")
        print(f"盲測交易次數: {len(final_oos_df)}")
        print("如果你只能用『過去』的參數來做『未來』的交易，這就是你的真實獲利。")
    else:
        print("無效的策略：在所有盲測期均未能產生獲利交易。")

# 啟動點
# === 啟動點：批量化橫截面 WFA 引擎 ===
# === 啟動點：批量化橫截面 WFA 引擎 ===
if __name__ == "__main__":
    LIST_FILE_PATH = "C:/Users/tame1/OneDrive/桌面/SMC-/stock_list.csv"
    
    try:
        # 🌟 修改 1：拿掉 header=None，因為你的檔案第一行有標題
        list_df = pd.read_csv(LIST_FILE_PATH) 
        
        # 🌟 修改 2：直接指定抓取 'ticker' 這個欄位
        target_tickers = list_df['ticker'].dropna().astype(str).tolist()
        
        print(f"✅ 成功載入 {len(target_tickers)} 檔標的準備進行 WFA 壓力測試！\n")
        print("="*50)
        
        # 開始自動循環每一檔股票
        for ticker in target_tickers:
            clean_ticker = ticker.strip() 
            run_wfa_for_stock(clean_ticker)
            print("\n" + "="*50 + "\n")
            
        print("🎉 所有標的 Walk Forward Analysis 執行完畢！")
        
    except FileNotFoundError:
        print(f"🚨 找不到股票清單檔案：{LIST_FILE_PATH}")
    except KeyError:
        print(f"🚨 CSV 裡面找不到 'ticker' 欄位，請確認你的欄位名稱沒有打錯（大小寫需一致）。")
    except Exception as e:
        print(f"🚨 執行批量測試時發生錯誤：{e}")