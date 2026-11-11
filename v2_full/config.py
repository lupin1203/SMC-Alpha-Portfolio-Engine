# config.py
import logging
import json
import datetime
import os

# ==========================================
# 結構化日誌設定
# ==========================================
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

class _StructuredFormatter(logging.Formatter):
    """輸出 JSON 格式日誌，方便事後查詢"""
    def format(self, record):
        log_obj = {
            "ts":      datetime.datetime.utcnow().isoformat() + "Z",
            "level":   record.levelname,
            "module":  record.module,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)

def setup_logging():
    """初始化結構化日誌，同時輸出到 console 和 JSON 檔"""
    root = logging.getLogger()
    if root.handlers:
        return  # 已初始化過，不重複設定

    root.setLevel(logging.INFO)

    # Console：普通文字格式
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root.addHandler(ch)

    # 每日 JSON 日誌檔
    today     = datetime.date.today().isoformat()
    json_path = os.path.join(LOG_DIR, f"smc_{today}.json")
    fh = logging.FileHandler(json_path, encoding='utf-8')
    fh.setFormatter(_StructuredFormatter())
    root.addHandler(fh)

    # 保留舊的 smc_system.log（相容原有設定）
    legacy = logging.FileHandler('smc_system.log', encoding='utf-8')
    legacy.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root.addHandler(legacy)

# 啟動時自動初始化
setup_logging()

# 預設台股名單 (防呆用)
DEFAULT_SEARCH_LIST = ["2330.TW - 台積電", "2317.TW - 鴻海", "2454.TW - 聯發科", "2603.TW - 長榮"]
DEFAULT_STOCK_NAMES = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2603.TW": "長榮"}
DEFAULT_INDUSTRY_DICT = {"權值股": ["2330.TW", "2317.TW", "2454.TW"], "航運": ["2603.TW"]}

# 交易摩擦力常數
FEE_RATE = 0.001425      # 手續費率
TAX_RATE = 0.003         # 證交稅率
DEFAULT_FEE_DISC = 0.28  # 預設手續費折讓
DEFAULT_SLIPPAGE = 0.002 # 預設滑價
DEFAULT_MIN_VOL = 1000   # 預設最低成交量
DEFAULT_RISK_PCT = 0.5   # 預設單筆風險 (%)，與實盤建議一致（小資金保守）
DEFAULT_CAPITAL = 1_000_000.0  # 預設初始本金