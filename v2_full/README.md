# v2_full — 2026 年 9 月完整版（封存）

這是本系統在 2026 年 9 月的完整版本，根目錄的程式是之後重新整理過的精簡研究版。
保留這份是為了記錄當時做過的功能：

- `app.py`：Streamlit 主介面（完整版）
- `daily_scheduler.py`、`update_all.py`：每日排程與資料更新
- `live_execution.py`、`telegram_bot_daemon.py`：Telegram 通知與遠端控制
- `revenue_fetcher.py`、`dynamic_sector.py`、`sector_radar.py`：營收篩選、族群輪動雷達
- `build_ml_features.py`、`train_model.py`：機器學習特徵與模型訓練
- `trade_journal.py`：交易日誌

金鑰從環境變數讀取：`TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`、`FINMIND_TOKEN`。

此版本不再維護；回測方式已在根目錄版本修正，結論以根目錄 README 為準。
