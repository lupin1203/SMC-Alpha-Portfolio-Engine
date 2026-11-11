# telegram_bot_daemon.py
import os
import time
import json
import logging
import requests
import threading # 🚀 新增：用來處理不卡死的背景任務
from live_execution import SystemWatchdog, send_telegram_message

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Telegram Bot Token 從環境變數讀取: $env:TELEGRAM_BOT_TOKEN="..."
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CONFIG_PATH = 'daily_config.json'

def load_central_config():
    """讀取中央設定檔，用完整預設值補齊缺少欄位"""
    defaults = {
        "gap_pct": 0.3,
        "min_vol": 1000,
        "risk_pct": 1.0,
        "account_dd_limit": 0.20,
        "sector_group": "ALL",
        "use_revenue_filter": True,
        "min_yoy": 15.0,
        "require_consecutive": True,
        "top_n_sectors": 3,
    }
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return cfg

def save_central_config(config_dict):
    """寫入中央設定檔"""
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def background_run_task():
    """獨立的背景執行任務，確保不卡死 Telegram 監聽器"""
    try:
        from daily_scheduler import daily_routine
        daily_routine()
        # 注意：不需要在這裡 return 文字，因為 daily_routine 跑完自己會發送報告到 TG
    except Exception as e:
        send_telegram_message(f"*背景掃描發生崩潰:* `{e}`")

def parse_and_execute_command(text):
    """解析手機傳來的文字指令"""
    text = text.strip()
    config = load_central_config()

    if text == '/status':
        rev_status = "開啟" if config.get('use_revenue_filter') else "❌ 關閉"
        consec = " 是" if config.get('require_consecutive') else "❌ 否"
        msg  = f"*【自動排程參數狀態】*\n\n"
        msg += f"*─ 掃描設定 ─*\n"
        msg += f"🔹 掃描範圍: `{config.get('sector_group', 'ALL')}`\n"
        msg += f"🔹 板塊數量: `{config.get('top_n_sectors', 3)} 個`\n"
        msg += f"🔹 最小缺口: `{config.get('gap_pct', 0.3)}%`\n"
        msg += f"🔹 最低成交量: `{config.get('min_vol', 1000)} 張`\n\n"
        msg += f"*─ 月營收篩選（第一道鎖）─*\n"
        msg += f"🔹 狀態: {rev_status}\n"
        msg += f"🔹 最低 YoY 門檻: `{config.get('min_yoy', 15.0)}%`\n"
        msg += f"🔹 要求連續兩月達標: {consec}\n\n"
        msg += f"*─ 風控設定 ─*\n"
        msg += f"🔹 單筆風險: `{config.get('risk_pct', 1.0)}%`\n"
        msg += f"🔹 帳戶停機線: `{config.get('account_dd_limit', 0.20)*100:.0f}%`\n\n"
        msg += f"💡 此參數供每日排程使用，不影響網頁端。"
        return msg

    elif text.startswith('/set_gap'):
        try:
            val = float(text.split()[1])
            config['gap_pct'] = val
            save_central_config(config)
            return f"*更新成功！*\n`gap_pct` → `{val}%`"
        except Exception:
            return "*格式錯誤！* 範例：`/set_gap 0.5`"

    elif text.startswith('/set_risk'):
        try:
            val = float(text.split()[1])
            config['risk_pct'] = val
            save_central_config(config)
            return f"*更新成功！*\n`risk_pct` → `{val}%`"
        except Exception:
            return "*格式錯誤！* 範例：`/set_risk 2.0`"

    elif text.startswith('/set_group'):
        try:
            val = text.split()[1].upper()
            valid = ['ALL', 'TECH', 'TRADITIONAL']
            if val not in valid:
                return f"支援的套餐：`ALL` / `TECH` / `TRADITIONAL`"
            config['sector_group'] = val
            save_central_config(config)
            return f"*更新成功！*\n掃描範圍 → `{val}`"
        except Exception:
            return "*格式錯誤！* 範例：`/set_group TECH`"

    elif text.startswith('/set_yoy'):
        try:
            val = float(text.split()[1])
            if val < 0:
                return "YoY 門檻不能為負數"
            config['min_yoy'] = val
            save_central_config(config)
            return f"*更新成功！*\n月營收 YoY 門檻 → `{val}%`"
        except Exception:
            return "*格式錯誤！* 範例：`/set_yoy 20.0`"

    elif text == '/rev_on':
        config['use_revenue_filter'] = True
        save_central_config(config)
        return f"*月營收篩選已開啟*（門檻 {config.get('min_yoy', 15.0)}%）"

    elif text == '/rev_off':
        config['use_revenue_filter'] = False
        save_central_config(config)
        return "*月營收篩選已關閉*，將直接用板塊雷達結果進行 SMC 掃描。"

    elif text.startswith('/set_sectors'):
        try:
            val = int(text.split()[1])
            if not (1 <= val <= 10):
                return "板塊數量需在 1~10 之間"
            config['top_n_sectors'] = val
            save_central_config(config)
            return f"*更新成功！*\nAI 雷達板塊數 → `{val} 個`"
        except Exception:
            return "*格式錯誤！* 範例：`/set_sectors 3`"

    elif text == '/help':
        return (
            "*指令清單*\n\n"
            "*─ 查詢 ─*\n"
            "`/status` 查看所有參數\n\n"
            "*─ 掃描設定 ─*\n"
            "`/set_group ALL|TECH|TRADITIONAL`\n"
            "`/set_sectors 3` 板塊數量\n"
            "`/set_gap 0.3` 最小缺口 (%)\n\n"
            "*─ 月營收篩選 ─*\n"
            "`/rev_on` 開啟第一道鎖\n"
            "`/rev_off` 關閉第一道鎖\n"
            "`/set_yoy 15.0` YoY 門檻 (%)\n\n"
            "*─ 風控 ─*\n"
            "`/set_risk 1.0` 單筆風險 (%)\n\n"
            "*─ 執行 ─*\n"
            "`/run` 立即觸發全市場掃描\n"
        )

    elif text == '/run':
        # 修復主執行緒卡死問題：將繁重的回測任務丟入背景 Thread
        send_telegram_message("*收到老闆密令！已將全市場掃描任務丟入背景全速運算中，請稍候...*")
        bg_thread = threading.Thread(target=background_run_task)
        bg_thread.start()
        return None # 直接返回 None，讓監聽器可以立刻繼續工作

    return None

def bot_polling_loop():
    """長輪詢主迴圈"""
    logging.info("Telegram ChatOps 守護進程已開機，正在背景監聽指令...")
    offset = 0
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    
    while True:
        try:
            response = requests.get(url, params={"offset": offset, "timeout": 20}, timeout=25)
            data = response.json()
            if "result" in data:
                for update in data["result"]:
                    offset = update["update_id"] + 1
                    if "message" in update and "text" in update["message"]:
                        raw_text = update["message"]["text"]
                        reply_msg = parse_and_execute_command(raw_text)
                        if reply_msg:
                            send_telegram_message(reply_msg)
        except Exception as e:
            time.sleep(5)

if __name__ == "__main__":
    watchdog = SystemWatchdog(max_retries=999) 
    watchdog.execute_with_retry("Telegram背景監聽守護任務", bot_polling_loop)