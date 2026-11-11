import os
import time
import datetime
import logging
import pandas as pd
import json
import subprocess
from live_execution import SystemWatchdog, send_telegram_message
from portfolio_engine import run_portfolio_backtest
from data_provider import load_data, fetch_stock_data
from sector_radar import SectorRadar
from revenue_fetcher import screen_sector_by_revenue
from dynamic_sector import get_sector_map
from trade_journal import load_live_trades, confirm_exit
from database import _get_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DEFAULT_CONFIG = {
    "gap_pct": 0.3,
    "min_vol": 1000,
    "risk_pct": 1.0,
    "account_dd_limit": 0.20,
    "sector_group": "ALL",
    "use_revenue_filter": True,
    "min_yoy": 15.0,
    "require_consecutive": True,
    "top_n_sectors": 3,
    "trailing_stop_pct": 0.10,
    "min_holding_days": 10,
    "catastrophic_stop_mult": 2.0,
    "regime_benchmark": "0050.TW",   # Regime 基準：0050.TW / TAIEX / TPEx
    "vol_threshold": 25.0,           # 高波動門檻 (%)
    # ── 即時掃描部位計算參數（與 Tab 1 一致）──
    "live_equity": 100000,          # 帳戶淨值
    "max_capital_per_trade": 0.20,  # 單筆資金上限
    "min_momentum": 2.0,            # 最低動能倍數
    "min_rr": 2.0,                  # 最低賠率
}

# ==========================================
# 持倉監控核心
# ==========================================

def _get_trading_days_held(entry_date_str: str) -> int:
    """計算進場至今的交易日數（近似：用自然日 × 0.71）"""
    try:
        entry_dt = pd.to_datetime(entry_date_str).date()
        calendar_days = (datetime.date.today() - entry_dt).days
        return int(calendar_days * 0.71)  # 台股一年約 252 交易日 / 365 自然日
    except Exception:
        return 999

def _update_position_high(trade_id: int, new_high: float):
    """更新持倉的歷史最高價（移動停利追蹤用）"""
    try:
        with _get_conn() as conn:
            c = conn.cursor()
            # 確保 position_high 欄位存在
            try:
                c.execute("ALTER TABLE live_trades ADD COLUMN position_high REAL DEFAULT 0")
            except Exception:
                pass  # 欄位已存在
            c.execute(
                "UPDATE live_trades SET position_high = MAX(COALESCE(position_high, 0), ?) WHERE id = ?",
                (new_high, trade_id)
            )
    except Exception as e:
        logging.warning(f"更新最高價失敗 trade_id={trade_id}: {e}")

def _get_position_high(trade_id: int) -> float:
    """取得持倉的歷史最高價"""
    try:
        with _get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT position_high FROM live_trades WHERE id = ?", (trade_id,))
            row = c.fetchone()
            return float(row[0]) if row and row[0] else 0.0
    except Exception:
        return 0.0

def monitor_positions(config: dict = None) -> list:
    """
    每日持倉監控主函數。
    
    檢查每一筆 open 持倉：
    1. 保護期狀態（前 N 天只掛緊急停損）
    2. 移動停利線是否被觸破
    3. Catastrophic_Stop 是否觸發
    4. 基本面是否惡化（每月 10 號）
    
    回傳需要發送的通知清單。
    """
    if config is None:
        config = _load_config()

    trailing_pct      = config.get('trailing_stop_pct', 0.10)
    min_holding_days  = config.get('min_holding_days', 10)
    cat_stop_mult     = config.get('catastrophic_stop_mult', 2.0)

    open_trades = load_live_trades(status='open')
    if open_trades.empty:
        logging.info("目前無持倉，跳過監控")
        return []

    notifications = []
    today = datetime.date.today()
    is_monthly_check = (today.day == 10)  # 每月 10 號跑基本面

    for _, trade in open_trades.iterrows():
        trade_id     = int(trade['id'])
        ticker       = trade['ticker']
        entry_price  = float(trade.get('actual_entry') or trade.get('signal_price') or 0)
        entry_date   = trade.get('entry_date') or trade.get('signal_date')
        stop_price   = float(trade.get('actual_stop') or trade.get('signal_stop') or 0)

        if entry_price <= 0 or not ticker:
            continue

        # ── 抓今日行情 ──
        try:
            df = fetch_stock_data(ticker, days=5)
            if df is None or df.empty:
                continue
            today_high  = float(df['High'].iloc[-1])
            today_close = float(df['Close'].iloc[-1])
            today_low   = float(df['Low'].iloc[-1])
        except Exception as e:
            logging.warning(f"抓行情失敗 {ticker}: {e}")
            continue

        # ── 更新歷史最高價 ──
        _update_position_high(trade_id, today_high)
        position_high = _get_position_high(trade_id)
        if position_high <= 0:
            position_high = entry_price  # 第一天，最高就是進場價

        # ── 計算停利線和停損距離 ──
        trailing_stop_line = position_high * (1 - trailing_pct)
        risk_dist = entry_price - stop_price if stop_price > 0 else entry_price * 0.05
        cat_stop_price = entry_price - (risk_dist * cat_stop_mult)

        # ── 保護期判斷 ──
        trading_days_held = _get_trading_days_held(str(entry_date)) if entry_date else 999
        in_protection = trading_days_held < min_holding_days

        # ── 保護期結束通知 ──
        if trading_days_held == min_holding_days:
            notifications.append({
                "type": "protection_end",
                "ticker": ticker,
                "msg": (
                    f"🔓 *{ticker} 保護期結束*\n"
                    f"已持倉 {trading_days_held} 個交易日，保護期滿。\n"
                    f"請將券商停損單從緊急停損價 `{cat_stop_price:.2f}` 調整為正常停損價 `{stop_price:.2f}`"
                )
            })

        # ── 出場判定 ──
        exit_triggered = False
        exit_reason    = ""
        exit_price     = 0.0

        if in_protection:
            # 保護期內：只判斷 Catastrophic_Stop
            if today_low <= cat_stop_price:
                exit_triggered = True
                exit_reason    = "Catastrophic_Stop"
                exit_price     = today_close  # 用收盤價（與回測一致）
                notifications.append({
                    "type": "exit_urgent",
                    "ticker": ticker,
                    "msg": (
                        f"🚨 *{ticker} 緊急停損觸發！*\n"
                        f"保護期內跌超過 {cat_stop_mult:.0f} 倍停損距離。\n"
                        f"今日收盤 `{today_close:.2f}` | 緊急停損價 `{cat_stop_price:.2f}`\n"
                        f"*建議明日開盤以市價賣出。*"
                    )
                })
        else:
            # 保護期結束：正常停損 + 移動停利
            if today_close <= trailing_stop_line:
                exit_triggered = True
                exit_reason    = "移動停利"
                exit_price     = today_close
                r_val = (today_close - entry_price) / risk_dist if risk_dist > 0 else 0
                notifications.append({
                    "type": "exit_trailing",
                    "ticker": ticker,
                    "msg": (
                        f"📈 *{ticker} 移動停利觸發*\n"
                        f"收盤價 `{today_close:.2f}` 跌破移動停利線 `{trailing_stop_line:.2f}`\n"
                        f"（歷史最高 `{position_high:.2f}` × {(1-trailing_pct)*100:.0f}%）\n"
                        f"估計 R 值：`{r_val:.2f} R`\n"
                        f"*建議明日開盤賣出。*"
                    )
                })
            elif today_low <= stop_price:
                exit_triggered = True
                exit_reason    = "Hit_Stop"
                exit_price     = stop_price
                notifications.append({
                    "type": "exit_stop",
                    "ticker": ticker,
                    "msg": (
                        f"🛑 *{ticker} 停損觸發*\n"
                        f"今日最低 `{today_low:.2f}` 跌破停損價 `{stop_price:.2f}`\n"
                        f"*券商停損單應已自動執行，請確認成交。*"
                    )
                })

        # ── 每月基本面檢查 ──
        if is_monthly_check and not exit_triggered:
            try:
                from revenue_fetcher import fetch_monthly_revenue_yoy, fetch_gross_margin
                ticker_pure = ticker.replace(".TW", "")
                rev = fetch_monthly_revenue_yoy(ticker_pure)
                gm  = fetch_gross_margin(ticker_pure)

                fund_issues = []
                if rev.get('data_ok') and rev.get('latest_yoy') is not None:
                    if rev['latest_yoy'] < 0:
                        fund_issues.append(f"月營收 YoY 轉負（{rev['latest_yoy']:.1f}%）")
                if gm.get('data_ok') and gm.get('gm_expanding') is False:
                    fund_issues.append("毛利率衰退")

                if fund_issues:
                    notifications.append({
                        "type": "fundamental_warning",
                        "ticker": ticker,
                        "msg": (
                            f"📊 *{ticker} 基本面警告（月度檢查）*\n"
                            f"問題：{' + '.join(fund_issues)}\n"
                            f"目前收盤 `{today_close:.2f}` | 進場價 `{entry_price:.2f}`\n"
                            f"*建議評估是否出場。*"
                        )
                    })
            except Exception as e:
                logging.warning(f"基本面檢查失敗 {ticker}: {e}")

        # ── 進度通知：每週五發一次持倉狀態摘要 ──
        if today.weekday() == 4 and not exit_triggered:  # 週五
            r_current = (today_close - entry_price) / risk_dist if risk_dist > 0 else 0
            protection_str = f"保護期（剩餘 {min_holding_days - trading_days_held} 交易日）" if in_protection else "正常監控"
            notifications.append({
                "type": "weekly_summary",
                "ticker": ticker,
                "msg": (
                    f"📋 *{ticker} 週報*\n"
                    f"狀態：{protection_str}\n"
                    f"進場 `{entry_price:.2f}` → 現在 `{today_close:.2f}`\n"
                    f"目前 R：`{r_current:.2f} R`\n"
                    f"移動停利線：`{trailing_stop_line:.2f}`\n"
                    f"歷史最高：`{position_high:.2f}`"
                )
            })

    return notifications


def run_position_monitor():
    """
    持倉監控入口，每日收盤後呼叫。
    把所有通知整理成一則 Telegram 訊息發出。
    """
    config = _load_config()
    notifications = monitor_positions(config)

    if not notifications:
        logging.info("持倉監控：今日無觸發事件")
        return

    # 按優先順序排序：緊急停損 > 移動停利 > 停損 > 基本面警告 > 保護期結束 > 週報
    priority = {
        "exit_urgent": 0,
        "exit_trailing": 1,
        "exit_stop": 2,
        "fundamental_warning": 3,
        "protection_end": 4,
        "weekly_summary": 5,
    }
    notifications.sort(key=lambda x: priority.get(x["type"], 9))

    # 緊急訊息立即單獨發送
    urgent = [n for n in notifications if n["type"] == "exit_urgent"]
    for n in urgent:
        send_telegram_message(n["msg"])

    # 其他訊息合併發送
    others = [n for n in notifications if n["type"] != "exit_urgent"]
    if others:
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        combined = f"📊 *【持倉監控日報 {today_str}】*\n共 {len(others)} 則通知\n\n"
        combined += "\n\n".join(n["msg"] for n in others)
        send_telegram_message(combined)

    logging.info(f"持倉監控完成，發送 {len(notifications)} 則通知")

def update_data_task():
    """執行完整資料更新（K線 + 基本面 + 板塊）。
    基本面有 7 天快取，每天跑也不會重複抓，等於自動判斷該不該更新財報。"""
    logging.info("開始執行每日完整資料更新 (update_all.py)...")
    try:
        result = subprocess.run(
            ["python", "update_all.py"],
            check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace",   # 強制 UTF-8，避免 Windows cp950 解碼錯誤
        )
        logging.info(result.stdout[-2000:] if result.stdout else "（無輸出）")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"資料更新腳本崩潰: {e.stderr}")
        # K線更新失敗才算致命；基本面/板塊限流不該擋住當天掃描
        # update_all 內部已有斷路器，這裡即使非零退出也讓掃描繼續（用本地快取）
        return True


def _load_config() -> dict:
    """讀取 daily_config.json，缺欄位用 DEFAULT_CONFIG 補齊"""
    try:
        with open('daily_config.json', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    # 用預設值補齊任何缺少的欄位
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg


def run_daily_scan():
    """
    執行全市場掃描，流程與 Tab 1 雷達完全一致：
      Step 0：判斷當前 Regime，不在允許環境就靜音
      Step 1：月營收板塊複合評分，選出最強 N 個板塊
      Step 2：月營收 YoY 個股篩選（第一道鎖）
      Step 3：SMC 訊號掃描，取今日進場訊號
    """
    config = _load_config()
    logging.info(f"掃描設定: {config}")

    INDUSTRY_DICT, _, _ = load_data()
    # ✅ 問題 4：板塊來源統一用 stock_list.csv（和 Tab 1 一致）
    # 只在 FinMind 快取不存在時才用 FinMind
    SECTOR_MAP = INDUSTRY_DICT if INDUSTRY_DICT else get_sector_map()

    # ── Step 0：Regime 判斷（與 Tab 1 一致）──
    allowed_regimes = ["Bull_LowVol", "Bear_HighVol"]
    try:
        _bench = config.get('regime_benchmark', '0050.TW')
        _vthr  = config.get('vol_threshold', 25.0)
        if _bench in ("TAIEX", "TPEx"):
            from revenue_fetcher import fetch_taiwan_index
            m0050 = fetch_taiwan_index(_bench, days=250)
        else:
            m0050 = fetch_stock_data("0050.TW", days=250)
        if m0050 is not None and not m0050.empty and len(m0050) > 200:
            close   = m0050['Close'].iloc[-1]
            sma200  = m0050['Close'].rolling(200).mean().iloc[-1]
            vol20   = m0050['Close'].pct_change().rolling(20).std().iloc[-1] * (252**0.5) * 100
            trend   = "Bull" if close > sma200 else "Bear"
            volreg  = "HighVol" if vol20 > _vthr else "LowVol"
            current_regime = f"{trend}_{volreg}"
        else:
            current_regime = "Unknown"
    except Exception:
        current_regime = "Unknown"

    logging.info(f"當前 Regime: {current_regime}")

    if current_regime == "Unknown":
        msg = (
            f"⚠️ *【SMC 每日掃描】*\n"
            f"無法判斷當前市場環境（基準資料不足）。\n"
            f"為求嚴謹，今日不發出任何進場訊號，保持空手。"
        )
        send_telegram_message(msg)
        logging.info("Regime Unknown，發送通知並跳過掃描")
        return ("notified", None)

    if current_regime not in allowed_regimes:
        msg = (
            f"🌡️ *【SMC 每日掃描】*\n"
            f"當前市場環境：`{current_regime}`\n"
            f"不在允許進場的環境（{' / '.join(allowed_regimes)}），今日系統靜音。\n"
            f"保持空手，等待環境改善。"
        )
        send_telegram_message(msg)
        logging.info(f"Regime 過濾：{current_regime} 不在允許清單，跳過掃描")
        return ("notified", None)

    group_mode = config['sector_group']
    if group_mode == 'TECH':
        active_sectors = ['半導體業', '電腦及週邊設備業', '光電業', '電子零組件業', '通信網路業']
    elif group_mode == 'TRADITIONAL':
        active_sectors = ['航運業', '鋼鐵工業', '塑膠工業', '建材營造', '電機機械']
    else:
        active_sectors = list(SECTOR_MAP.keys())

    candidate_tickers = list({t for sec in active_sectors for t in SECTOR_MAP.get(sec, [])})
    logging.info(f"[{group_mode}] 候選池：{len(candidate_tickers)} 檔")

    # ── Step 1：月營收板塊複合評分（與 Tab 1 一致）──
    raw_data_dict = {}
    try:
        from concurrent.futures import ThreadPoolExecutor
        import os
        DATA_DIR = "./local_data"
        def _read(t):
            sid = str(t).replace(".TW.TW","").replace(".TW","")
            path = os.path.join(DATA_DIR, f"{sid}.TW.parquet")
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    return t, df
                except Exception:
                    pass
            return t, None
        with ThreadPoolExecutor(max_workers=8) as ex:
            for t, df in ex.map(_read, candidate_tickers):
                if df is not None:
                    raw_data_dict[t] = df
    except Exception as e:
        logging.warning(f"K 線預載失敗: {e}")

    radar = SectorRadar(SECTOR_MAP)
    # 取得最強板塊清單（不只標的，還要知道是哪些板塊）
    radar_scores = radar._score_composite(raw_data_dict, min_yoy=config.get('min_yoy', 10.0))
    top_sectors  = [s for s, _ in sorted(radar_scores.items(), key=lambda x: x[1], reverse=True)[:config['top_n_sectors']]]
    if not top_sectors:
        top_sectors = active_sectors[:config['top_n_sectors']]
    target_tickers = list({t for sec in top_sectors for t in SECTOR_MAP.get(sec, [])})
    logging.info(f"Step 1 完成：最強板塊 {top_sectors}，{len(target_tickers)} 檔")

    # ── Step 2：月營收 YoY 篩選（記錄通過/未通過，不直接踢除）──
    revenue_pass_set = set()
    if config['use_revenue_filter'] and target_tickers:
        logging.info(f"Step 2：月營收篩選（YoY ≥ {config['min_yoy']}%）")
        try:
            rev_df = screen_sector_by_revenue(
                target_tickers,
                min_yoy=config['min_yoy'],
                require_consecutive=config['require_consecutive'],
                require_gm_expanding=False,
                cache_only=True,
            )
            revenue_pass_set = set(rev_df[rev_df['pass_filter'] == True]['ticker'].tolist())
            logging.info(f"Step 2 完成：{len(revenue_pass_set)} 檔通過月營收篩選")
        except Exception as e:
            logging.warning(f"月營收篩選失敗: {e}")

    # ── Step 3：逐檔即時掃描 FVG（與 Tab 1 完全一致）──
    from strategy import apply_smc_logic
    from revenue_fetcher import fetch_gross_margin
    from utils import safe_divide

    live_equity   = config.get('live_equity', 100000)
    live_risk_pct = config['risk_pct'] / 100.0
    trailing_pct  = config.get('trailing_stop_pct', 0.10)
    max_cap_pct   = config.get('max_capital_per_trade', 0.20)
    min_momentum  = config.get('min_momentum', 2.0)
    min_rr        = config.get('min_rr', 2.0)

    sector_reverse = {t: sec for sec in top_sectors for t in SECTOR_MAP.get(sec, [])}
    results = []
    seen_sectors = {}
    min_vol_abs = config.get('min_vol', 1000)  # 絕對最低日均成交量（張）

    for ticker in target_tickers:
        df = raw_data_dict.get(ticker)
        if df is None or len(df) < 65:
            continue
        try:
            df_logic = apply_smc_logic(df.copy(), min_gap_pct=config['gap_pct'], use_sma_filter=True)
            last     = df_logic.iloc[-1]
            prev     = df_logic.iloc[-2] if len(df_logic) >= 2 else last

            # ── 問題 1 修正：絕對成交量門檻 ──
            vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
            if vol_ma20 < min_vol_abs:
                logging.debug(f"{ticker} 日均成交量 {vol_ma20:.0f} 張 < {min_vol_abs} 張，跳過")
                continue

            if not last.get('FVG', False):
                continue

            # ── 問題 2 修正：FVG 確認條件 ──
            # 缺口下緣（前兩天高點），今天收盤要在缺口上方（缺口未被完全回補）
            gap_lower = prev['High'] if 'High' in prev else 0
            if last['Close'] < gap_lower:
                logging.debug(f"{ticker} FVG 缺口已被回補，跳過")
                continue

            vol_surge  = safe_divide(df['Volume'].iloc[-1], vol_ma20)
            entry      = last['Low']
            sl         = entry - last['adjusted_risk']
            risk_dist  = max(entry - sl, 0.01)

            # ── 問題 3 修正：移動停利加最低 1R 啟動門檻 ──
            # 獲利目標 = 至少 1R 以上才啟動移動停利
            min_profit_before_trailing = entry + risk_dist  # 1R
            tp_trailing = entry * (1 + trailing_pct)
            tp = max(tp_trailing, min_profit_before_trailing)
            rr = safe_divide(tp - entry, risk_dist)

            if vol_surge < min_momentum or rr < min_rr:
                continue

            # ── 問題 4 修正：板塊來源統一用 stock_list.csv ──
            # sector_reverse 已在 Step 1 改用 INDUSTRY_DICT 建立（見下方）
            ticker_sector = sector_reverse.get(ticker, "其他")

            # 相關性過濾：同板塊最多 2 檔
            if seen_sectors.get(ticker_sector, 0) >= 2:
                continue

            # 部位計算
            risk_amt   = live_equity * live_risk_pct
            risk_share = risk_dist
            shares     = int(risk_amt / risk_share)
            cap_req    = shares * entry
            max_cash   = live_equity * max_cap_pct
            if cap_req > max_cash:
                shares  = int(max_cash / entry)
                cap_req = shares * entry
            if shares <= 0:
                continue

            # 月營收狀態
            rev_status = "✅ 通過" if ticker in revenue_pass_set else "⚠️ 未通過"

            # 財報（毛利率）狀態
            try:
                gm = fetch_gross_margin(ticker.replace(".TW", ""), cache_only=True)
                if gm.get('data_ok'):
                    gm_status = "✅ 擴張" if gm.get('gm_expanding') else "⚠️ 衰退"
                else:
                    gm_status = "❓ 無資料"
            except Exception:
                gm_status = "❓ 無資料"

            seen_sectors[ticker_sector] = seen_sectors.get(ticker_sector, 0) + 1
            results.append({
                "標的":      ticker,
                "板塊":      ticker_sector,
                "動能評分":  round(vol_surge, 2),
                "預期賠率":  round(rr, 1),
                "進場價":    round(entry, 2),
                "防彈止損價": round(sl, 2),
                "獲利目標":  round(tp, 2),
                "建議股數":  shares,
                "預估耗資":  round(cap_req, 0),
                "月營收":    rev_status,
                "財報毛利":  gm_status,
            })
        except Exception as e:
            logging.debug(f"掃描 {ticker} 失敗: {e}")
            continue

    if not results:
        return ("ok", {
            "date": datetime.date.today().isoformat(),
            "data": pd.DataFrame(),
            "top_sectors": top_sectors,
            "regime": current_regime,
        })

    return ("ok", {
        "date": datetime.date.today().isoformat(),
        "data": pd.DataFrame(results),
        "top_sectors": top_sectors,
        "regime": current_regime,
    })

def format_telegram_msg(scan_result):
    """將掃描結果排版成適合手機閱讀的格式"""
    if scan_result is None:
        return "📊 *【每日 SMC 掃描報告】*\n今日無掃描結果。"

    top_sectors = scan_result.get("top_sectors", [])
    regime      = scan_result.get("regime", "Unknown")
    real_date   = scan_result.get("date", "")
    signals_df  = scan_result.get("data")

    if signals_df is None or signals_df.empty:
        msg  = f"📊 *【每日 SMC 掃描報告】*\n"
        msg += f"📅 數據日期: `{real_date}`\n"
        msg += f"🌡️ 市場環境: `{regime}`\n"
        msg += f"🏆 最強板塊: {' / '.join(top_sectors) if top_sectors else '無'}\n\n"
        msg += "今日這些板塊內無符合條件的進場訊號，保持空手，等待獵物。"
        return msg

    msg  = f"📊 *【SMC 獵物掃描報告】*\n"
    msg += f"📅 數據日期: `{real_date}`\n"
    msg += f"🌡️ 市場環境: `{regime}`\n"
    msg += f"🏆 最強板塊: {' / '.join(top_sectors)}\n"
    msg += f"🎯 共發現 {len(signals_df)} 檔進場訊號：\n\n"

    for _, row in signals_df.iterrows():
        msg += f"🔹 *{row['標的']}* ({row['板塊']})\n"
        msg += f"   動能評分: `{row['動能評分']:.2f}x` | 賠率: `{row['預期賠率']:.1f} R`\n"
        msg += f"   進場價: `{row['進場價']:.2f}`\n"
        msg += f"   防彈止損: `{row['防彈止損價']:.2f}`\n"
        msg += f"   獲利目標: `{row['獲利目標']:.2f}`\n"
        msg += f"   建議股數: `{row['建議股數']}` 股\n"
        msg += f"   預估耗資: `${row['預估耗資']:,.0f}`\n"
        msg += f"   月營收: {row['月營收']} | 財報毛利: {row['財報毛利']}\n"
        msg += "------------------------\n"

    msg += "\n⚠️ *提醒：月營收/財報「未通過」的標的，是技術面訊號但基本面較弱，請自行評估。*"
    msg += "\n⚠️ *請於明日開盤前設定好觸價單與緊急停損單！*"
    return msg

def daily_routine():
    """每日主流程。保證每天一定發出一則 Telegram 訊息。"""
    watchdog = SystemWatchdog(max_retries=3)

    # 1. 確保資料最新且無污染
    update_success = watchdog.execute_with_retry("每日資料池更新", update_data_task)
    if not update_success:
        send_telegram_message("❌ 每日資料更新發生嚴重崩潰，為保護資金，今日掃描任務已緊急終止。")
        return

    # 2. 持倉監控（優先於掃描，確保現有持倉安全）
    watchdog.execute_with_retry("每日持倉監控", run_position_monitor)

    # 3. 執行掃描
    #    run_daily_scan 在 Regime 靜音/Unknown 時自己發訊息並回傳 ("notified", None)
    #    正常掃描完回傳 ("ok", scan_result)
    scan_outcome = watchdog.execute_with_retry("全市場選股掃描", run_daily_scan)

    # 4. 發送報告
    try:
        if scan_outcome is None:
            # 掃描整個崩潰（watchdog 重試後仍失敗）→ 保底通知
            send_telegram_message("⚠️ *【SMC 每日掃描】*\n今日掃描未完成，系統保持空手。請檢查日誌。")
            return
        status, scan_result = scan_outcome
        if status == "notified":
            # Regime 靜音/Unknown，run_daily_scan 已自行發過訊息，不重複發
            return
        telegram_msg = format_telegram_msg(scan_result)
        watchdog.execute_with_retry("發送 Telegram 報告", send_telegram_message, telegram_msg)
    except Exception as e:
        logging.error(f"發送每日報告失敗: {e}")
        send_telegram_message("⚠️ 每日報告產生時發生錯誤，請檢查系統日誌。")

if __name__ == "__main__":
    daily_routine()