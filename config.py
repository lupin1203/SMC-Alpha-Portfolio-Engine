# config.py
import logging

# 系統日誌設定
logging.basicConfig(
    filename='smc_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 預設台股名單 (防呆用)
DEFAULT_SEARCH_LIST = ["2330.TW - 台積電", "2317.TW - 鴻海", "2454.TW - 聯發科", "2603.TW - 長榮"]
DEFAULT_STOCK_NAMES = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2603.TW": "長榮"}
DEFAULT_INDUSTRY_DICT = {"權值股": ["2330.TW", "2317.TW", "2454.TW"], "航運": ["2603.TW"]}

# 交易摩擦力常數
FEE_RATE = 0.001425  # 手續費率
TAX_RATE = 0.003     # 證交稅率