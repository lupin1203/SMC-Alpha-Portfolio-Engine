"""
update_all.py  —  一鍵更新所有資料
執行：python update_all.py

流程：
  1. 清理毒化的基本面快取（抓失敗的舊結果）
  2. 更新全市場 K 線（增量，只補新的）
  3. 更新月營收 + 毛利率歷史序列
  4. 更新板塊地圖（FinMind）
  5. 全部完成後回報統計

設計原則：
  - 增量更新，不砍舊資料（要砍掉重練請用 nuclear_update.py）
  - 基本面用 7 天快取，不會每天重打 FinMind
  - 每一步獨立 try/except，單一步驟失敗不影響其他步驟
"""
import os
import sys
import time
import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 強制 UTF-8 輸出（Windows 中文系統防呆）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


def step_0_clean_poisoned_cache():
    """清掉抓失敗的基本面快取，避免毒化"""
    logging.info("=" * 50)
    logging.info("[Step 0] 清理毒化的基本面快取...")
    cleaned = 0
    cache_files = [
        "revenue_cache/monthly_revenue.json",
        "revenue_cache/gross_margin.json",
        "revenue_cache/cashflow_cache.json",
    ]
    import json
    for cf in cache_files:
        if not os.path.exists(cf):
            continue
        try:
            with open(cf, encoding='utf-8') as f:
                cache = json.load(f)
            # 移除 data_ok=False 的毒化項目
            before = len(cache)
            cache = {k: v for k, v in cache.items()
                     if not (isinstance(v, dict) and v.get("data_ok") is False)}
            after = len(cache)
            removed = before - after
            if removed > 0:
                with open(cf, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, ensure_ascii=False)
                cleaned += removed
                logging.info(f"  {cf}: 清掉 {removed} 筆失敗快取")
        except Exception as e:
            logging.warning(f"  {cf} 清理失敗: {e}")
    logging.info(f"[Step 0] 完成，共清理 {cleaned} 筆毒化快取")


def step_1_update_kline():
    """更新全市場 K 線（增量）"""
    logging.info("=" * 50)
    logging.info("[Step 1] 更新全市場 K 線（增量）...")
    try:
        from update_data import force_merge_and_purge_ghost_files, stage_2_download_all_stocks
        force_merge_and_purge_ghost_files(stage_label="下載前淨化")
        stage_2_download_all_stocks(years=10)
        force_merge_and_purge_ghost_files(stage_label="下載後決殺")
        logging.info("[Step 1] K 線更新完成")
        return True
    except Exception as e:
        logging.error(f"[Step 1] K 線更新失敗: {e}")
        return False


def step_2_update_fundamentals():
    """更新月營收 + 毛利率歷史序列"""
    logging.info("=" * 50)
    logging.info("[Step 2] 更新基本面歷史序列...")
    try:
        from data_provider import load_data
        from revenue_fetcher import build_fundamental_history_cache
        INDUSTRY_DICT, _, _ = load_data(use_local_scan=False)
        all_tickers = list({t for tks in INDUSTRY_DICT.values() for t in tks})
        logging.info(f"  共 {len(all_tickers)} 檔需更新（7天內快取會跳過）")
        build_fundamental_history_cache(all_tickers, delay=0.4)
        logging.info("[Step 2] 基本面更新完成")
        return True
    except Exception as e:
        logging.error(f"[Step 2] 基本面更新失敗: {e}")
        return False


def step_3_update_sector_map():
    """更新板塊地圖（FinMind）。限流時不覆蓋現有快取。"""
    logging.info("=" * 50)
    logging.info("[Step 3] 更新板塊地圖...")
    try:
        from dynamic_sector import fetch_dynamic_sector_map, get_sector_map
        # 先試抓新的（不存檔），fetch 內部已做正規化。確認有資料才覆蓋快取
        new_map = fetch_dynamic_sector_map(save_cache=False)
        if new_map and len(new_map) >= 10:
            import json
            with open('sector_map_cache.json', 'w', encoding='utf-8') as f:
                json.dump(new_map, f, ensure_ascii=False, indent=4)
            logging.info(f"[Step 3] 板塊地圖更新完成，共 {len(new_map)} 個板塊")
            return True
        else:
            # 限流或抓失敗，保留舊快取不覆蓋
            old_map = get_sector_map(force_update=False)
            logging.warning(f"[Step 3] FinMind 板塊抓取失敗（可能限流），保留現有快取（{len(old_map)} 個板塊）")
            return False
    except Exception as e:
        logging.error(f"[Step 3] 板塊地圖更新失敗: {e}")
        return False


if __name__ == "__main__":
    t_start = time.time()
    logging.info("一鍵更新所有資料 開始")

    step_0_clean_poisoned_cache()
    ok1 = step_1_update_kline()
    ok2 = step_2_update_fundamentals()
    ok3 = step_3_update_sector_map()

    elapsed = (time.time() - t_start) / 60
    logging.info("=" * 50)
    logging.info("全部更新完成總結：")
    logging.info(f"  K 線:     {'成功' if ok1 else '失敗'}")
    logging.info(f"  基本面:   {'成功' if ok2 else '失敗'}")
    logging.info(f"  板塊地圖: {'成功' if ok3 else '失敗'}")
    logging.info(f"  總耗時:   {elapsed:.1f} 分鐘")
    logging.info("=" * 50)