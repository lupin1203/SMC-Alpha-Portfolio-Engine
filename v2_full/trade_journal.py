# trade_journal.py
"""
實盤對帳機制
記錄每筆實際成交，與系統訊號比對，追蹤滑價差異
"""
import datetime
import logging
import pandas as pd
from database import _get_conn

def init_journal_db():
    """建立實盤日誌資料表"""
    try:
        with _get_conn() as conn:
            c = conn.cursor()
            # 實盤交易記錄
            c.execute('''CREATE TABLE IF NOT EXISTS live_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_date TEXT,        -- 系統訊號日期
                entry_date TEXT,         -- 實際進場日期
                ticker TEXT,
                sector TEXT,
                signal_price REAL,       -- 系統訊號進場價
                actual_entry REAL,       -- 實際成交價
                signal_stop REAL,        -- 系統止損價
                actual_stop REAL,        -- 實際掛單止損價
                shares INTEGER,
                status TEXT DEFAULT 'open',  -- open / closed / cancelled
                exit_date TEXT,
                signal_exit_price REAL,
                actual_exit REAL,
                exit_reason TEXT,
                signal_r REAL,           -- 系統預期 R
                actual_r REAL,           -- 實際 R
                slippage_entry_pct REAL, -- 進場滑價 %
                slippage_exit_pct REAL,  -- 出場滑價 %
                notes TEXT,
                created_at TEXT
            )''')
            # 參數變更日誌
            c.execute('''CREATE TABLE IF NOT EXISTS param_changelog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                changed_at TEXT,
                param_name TEXT,
                old_value TEXT,
                new_value TEXT,
                reason TEXT
            )''')
    except Exception as e:
        logging.error(f"實盤日誌初始化失敗: {e}")


def log_signal_entry(ticker, sector, signal_date, signal_price, signal_stop, shares, notes=""):
    """記錄系統發出的進場訊號（尚未實際成交）"""
    try:
        with _get_conn() as conn:
            conn.cursor().execute(
                '''INSERT INTO live_trades
                   (signal_date, ticker, sector, signal_price, signal_stop, shares, status, created_at, notes)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)''',
                (signal_date, ticker, sector, signal_price, signal_stop, shares,
                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), notes)
            )
        return True
    except Exception as e:
        logging.error(f"記錄訊號失敗: {e}")
        return False


def confirm_entry(trade_id, actual_entry, actual_stop, entry_date=None):
    """確認實際進場，記錄實際成交價與滑價"""
    try:
        with _get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT signal_price FROM live_trades WHERE id=?", (trade_id,))
            row = c.fetchone()
            if not row:
                return False
            signal_price = row[0]
            slippage_pct = round((actual_entry - signal_price) / signal_price * 100, 3) if signal_price else 0
            entry_date   = entry_date or datetime.date.today().isoformat()
            c.execute(
                '''UPDATE live_trades SET
                   actual_entry=?, actual_stop=?, entry_date=?,
                   slippage_entry_pct=?, status='open'
                   WHERE id=?''',
                (actual_entry, actual_stop, entry_date, slippage_pct, trade_id)
            )
        return True
    except Exception as e:
        logging.error(f"確認進場失敗: {e}")
        return False


def confirm_exit(trade_id, actual_exit, exit_reason, exit_date=None):
    """確認實際出場，計算實際 R 值與出場滑價"""
    try:
        with _get_conn() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT actual_entry, actual_stop, signal_price, signal_stop, shares FROM live_trades WHERE id=?",
                (trade_id,)
            )
            row = c.fetchone()
            if not row:
                return False
            actual_entry, actual_stop, signal_price, signal_stop, shares = row

            risk_per_share = (actual_entry - actual_stop) if actual_stop else 1
            actual_r = round((actual_exit - actual_entry) / risk_per_share, 3) if risk_per_share != 0 else 0

            signal_risk = (signal_price - signal_stop) if signal_stop else 1
            signal_r    = round((actual_exit - signal_price) / signal_risk, 3) if signal_risk != 0 else 0

            slippage_exit = round((actual_exit - signal_price) / signal_price * 100, 3) if signal_price else 0
            exit_date     = exit_date or datetime.date.today().isoformat()

            c.execute(
                '''UPDATE live_trades SET
                   actual_exit=?, exit_reason=?, exit_date=?,
                   actual_r=?, signal_r=?, slippage_exit_pct=?, status='closed'
                   WHERE id=?''',
                (actual_exit, exit_reason, exit_date, actual_r, signal_r, slippage_exit, trade_id)
            )
        return True
    except Exception as e:
        logging.error(f"確認出場失敗: {e}")
        return False


def log_param_change(param_name, old_value, new_value, reason=""):
    """記錄參數變更"""
    try:
        with _get_conn() as conn:
            conn.cursor().execute(
                '''INSERT INTO param_changelog (changed_at, param_name, old_value, new_value, reason)
                   VALUES (?, ?, ?, ?, ?)''',
                (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 param_name, str(old_value), str(new_value), reason)
            )
        return True
    except Exception as e:
        logging.error(f"記錄參數變更失敗: {e}")
        return False


def load_live_trades(status=None):
    """讀取實盤交易記錄"""
    try:
        with _get_conn() as conn:
            if status:
                return pd.read_sql_query(
                    "SELECT * FROM live_trades WHERE status=? ORDER BY id DESC", conn, params=(status,))
            return pd.read_sql_query("SELECT * FROM live_trades ORDER BY id DESC", conn)
    except Exception:
        return pd.DataFrame()


def load_param_changelog():
    """讀取參數變更歷史"""
    try:
        with _get_conn() as conn:
            return pd.read_sql_query("SELECT * FROM param_changelog ORDER BY id DESC", conn)
    except Exception:
        return pd.DataFrame()


def get_slippage_summary():
    """
    彙總實際滑價統計，與回測預設值比較
    回傳 dict: avg_entry_slippage, avg_exit_slippage, signal_vs_actual_r
    """
    df = load_live_trades(status='closed')
    if df.empty:
        return {}
    return {
        "avg_entry_slippage_pct": round(df['slippage_entry_pct'].mean(), 3),
        "avg_exit_slippage_pct":  round(df['slippage_exit_pct'].mean(), 3),
        "avg_signal_r":           round(df['signal_r'].mean(), 3),
        "avg_actual_r":           round(df['actual_r'].mean(), 3),
        "r_decay":                round(df['signal_r'].mean() - df['actual_r'].mean(), 3),
        "total_closed":           len(df),
    }