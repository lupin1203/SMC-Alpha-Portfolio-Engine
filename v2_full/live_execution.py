import os
import time
import logging
import requests
from requests.exceptions import RequestException

# 設定日誌格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 📱 Telegram API 設定區
# ==========================================
# 🔒 從環境變數讀取，不寫死在程式碼裡
#    設定方式（PowerShell）：
#    $env:TELEGRAM_BOT_TOKEN="你的新token"
#    $env:TELEGRAM_CHAT_ID="你的chat_id"
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def send_telegram_message(message):
    """發送 Telegram 訊息的共用小工具"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"\n📱 [模擬手機震動 — 未設定 TELEGRAM 環境變數] {message}\n")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        logging.error(f"Telegram 發送失敗: {e}")

# ==========================================
# 🛡️ 實盤看門狗系統
# ==========================================
class SystemWatchdog:
    def __init__(self, max_retries=3, base_delay=2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.is_alive = True  

    def execute_with_retry(self, task_name, func, *args, **kwargs):
        if not self.is_alive:
            logging.error(f"[拒絕執行 {task_name}] 系統已熔斷！")
            return None

        retries = 0
        while retries <= self.max_retries:
            try:
                logging.info(f"正在執行: {task_name}...")
                result = func(*args, **kwargs)
                logging.info(f"{task_name} 執行成功！")
                return result

            except RequestException as e:
                retries += 1
                if retries > self.max_retries:
                    logging.critical(f"[{task_name}] 連續 {self.max_retries} 次請求失敗: {e}")
                    self._trigger_kill_switch()
                    return None
                
                wait_time = self.base_delay ** retries
                logging.warning(f"[{task_name}] 網路異常，{wait_time} 秒後進行重試...")
                time.sleep(wait_time)

            except Exception as e:
                logging.critical(f"[{task_name}] 發生未知致命錯誤: {e}")
                self._trigger_kill_switch()
                return None

    def _trigger_kill_switch(self):
        self.is_alive = False
        logging.critical("[FATAL] 看門狗已觸發熔斷機制！系統強制鎖死！")
        
        # 🔔 真實觸發 Telegram 警報！
        alert_msg = " *【量化系統嚴重警報】*\n系統發生致命錯誤或連續斷線，已自動觸發**熔斷機制**！\n請立即開啟遠端桌面檢查伺服器與部位狀況！"
        send_telegram_message(alert_msg)

# 🧪 測試用
if __name__ == "__main__":
    # 測試發送一般訊息
    send_telegram_message("量化拷問終端 V20.0 連線測試成功！")
    
    # 測試看門狗熔斷
    watchdog = SystemWatchdog(max_retries=3)
    def broken_task():
        requests.get("https://this-website-does-not-exist.com.tw", timeout=3)
    watchdog.execute_with_retry("券商下單測試", broken_task)