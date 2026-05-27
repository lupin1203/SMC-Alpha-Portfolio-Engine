# database.py
import sqlite3
import datetime
import pandas as pd
import logging

DB_NAME = "smc_system.db"

def init_db():
    """初始化資料庫與資料表"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS backtest_history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      timestamp TEXT, ticker TEXT, gap REAL, tp_val REAL,
                      expectancy REAL, win_rate REAL, max_dd REAL, total_r REAL)''')
        conn.commit()
    except Exception as e:
        logging.error(f"資料庫初始化失敗: {str(e)}")
    finally:
        conn.close()

def save_backtest_record(ticker, gap, tp_val, exp, win_rate, max_dd, total_r):
    """儲存單筆回測紀錄"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''INSERT INTO backtest_history 
                     (timestamp, ticker, gap, tp_val, expectancy, win_rate, max_dd, total_r) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (now, ticker, gap, tp_val, exp, win_rate, max_dd, total_r))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"儲存紀錄失敗: {str(e)}")
        return False
    finally:
        conn.close()

def save_backtest_records_batch(records_list):
    """
    🔴 核心優化：一次寫入多筆回測紀錄 (Batch I/O)
    records_list 是一個 list of dicts: 
    [{'ticker': '2330', 'gap': 1.5, 'tp_val': 3.0, 'exp': 1.2, 'win_rate': 0.6, 'max_dd': 10.5, 'total_r': 50.2}, ...]
    """
    if not records_list:
        return False
        
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 準備資料 Tuples
        data_to_insert = [
            (now, r['ticker'], r['gap'], r['tp_val'], r['exp'], r['win_rate'], r['max_dd'], r['total_r']) 
            for r in records_list
        ]
        
        # 使用 executemany 進行批次寫入，大幅降低硬碟 I/O
        c.executemany('''INSERT INTO backtest_history 
                         (timestamp, ticker, gap, tp_val, expectancy, win_rate, max_dd, total_r) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', data_to_insert)
        
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"批次儲存紀錄失敗: {str(e)}")
        return False
    finally:
        conn.close()

def load_history():
    """讀取歷史紀錄"""
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM backtest_history ORDER BY id DESC", conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()