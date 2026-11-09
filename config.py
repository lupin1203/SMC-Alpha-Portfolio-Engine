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

# ==========================================
# 研究紀律：考試資料 (Holdout)
# 這個日期之後的資料保留作為「最後考試」。所有參數調整只能用之前的資料 (研究期)，
# 全部定案後才用考試期驗證一次；看過考試結果後再改參數，這段資料就失效了。
# ==========================================
HOLDOUT_START = "2024-01-01"
