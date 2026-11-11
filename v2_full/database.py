# database.py
import sqlite3
import datetime
import pandas as pd
import logging
from contextlib import contextmanager

DB_NAME = "smc_system.db"

@contextmanager
def _get_conn():
    """統一的連線管理器，確保任何情況下都正確關閉連線"""
    conn = sqlite3.connect(DB_NAME)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_db():
    """初始化資料庫與所有資料表"""
    try:
        with _get_conn() as conn:
            conn.cursor().execute('''CREATE TABLE IF NOT EXISTS backtest_history
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          timestamp TEXT, ticker TEXT, gap REAL, tp_val REAL,
                          expectancy REAL, win_rate REAL, max_dd REAL, total_r REAL)''')
        # 同步初始化實盤日誌表
        from trade_journal import init_journal_db
        init_journal_db()
    except Exception as e:
        logging.error(f"資料庫初始化失敗: {str(e)}")

def save_backtest_record(ticker, gap, tp_val, exp, win_rate, max_dd, total_r):
    """儲存單筆回測紀錄"""
    try:
        with _get_conn() as conn:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.cursor().execute(
                '''INSERT INTO backtest_history 
                   (timestamp, ticker, gap, tp_val, expectancy, win_rate, max_dd, total_r) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (now, ticker, gap, tp_val, exp, win_rate, max_dd, total_r)
            )
        return True
    except Exception as e:
        logging.error(f"儲存紀錄失敗: {str(e)}")
        return False

def save_backtest_records_batch(records_list):
    """
    核心優化：一次寫入多筆回測紀錄 (Batch I/O)
    records_list 是一個 list of dicts:
    [{'ticker': '2330', 'gap': 1.5, 'tp_val': 3.0, 'exp': 1.2, 'win_rate': 0.6, 'max_dd': 10.5, 'total_r': 50.2}, ...]
    """
    if not records_list:
        return False
    try:
        with _get_conn() as conn:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_to_insert = [
                (now, r['ticker'], r['gap'], r['tp_val'], r['exp'], r['win_rate'], r['max_dd'], r['total_r'])
                for r in records_list
            ]
            conn.cursor().executemany(
                '''INSERT INTO backtest_history 
                   (timestamp, ticker, gap, tp_val, expectancy, win_rate, max_dd, total_r) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                data_to_insert
            )
        return True
    except Exception as e:
        logging.error(f"批次儲存紀錄失敗: {str(e)}")
        return False

def load_history():
    """讀取歷史紀錄"""
    try:
        with _get_conn() as conn:
            return pd.read_sql_query("SELECT * FROM backtest_history ORDER BY id DESC", conn)
    except Exception:
        return pd.DataFrame()