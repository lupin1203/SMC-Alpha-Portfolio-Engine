# VCP + 族群連動（研究模組）

計畫：`vcp_sector_plan.md`。目前完成 **Phase 1–3（資料層、族群分群、族群強度）**，
VCP（Phase 5–6）與回測（Phase 7–8）尚未開始。

## 執行（在 repo 根目錄）

```bash
python -m pip install pandas numpy pyyaml plotly pyarrow pytest requests yfinance
python -m research.vcp_sector.report                 # 官方產業別
python -m research.vcp_sector.report --groups theme  # groups.yaml 自訂題材
python -m research.vcp_sector.report --synthetic     # 人造資料，只檢查流程
python -m pytest research/vcp_sector/tests -q
```

輸出在 `research/vcp_sector/output/`（CSV + 熱力圖 HTML，已 gitignore）。

## 資料來源（全部是還原權息價）
1. 既有 `local_data/{id}.parquet`（SMC 系統自己維護，照用；太舊會出 warning）
2. `research/vcp_sector/cache/prices_adj/`（本模組快取，落後 3 個營業日以上自動重抓）
3. yfinance 批次下載（auto_adjust，先 `.TW` 再 `.TWO`）—— 全市場第一次約 2–3 分鐘
4. FinMind `TaiwanStockPriceAdj`（需贊助會員；免費帳號被拒一次後就不再呼叫）
- **不使用未還原價**：除權息跳空會被當成下跌，污染均線與報酬。
- **盤中不完整的今日 K 棒會被丟掉**（台北 14:30 前），也不會寫進快取。
- 大盤：yfinance `^TWII` → FinMind `TAIEX`（指數免費帳號可抓）→ `0050`。
  TAIEX 是價格指數（不含息），group_rs 會系統性偏高約殖利率幅度；對所有族群相同，不影響排名。
- 產業分類：FinMind `TaiwanStockInfo`，取不到退回 `stock_list.csv`（`未分類 (35–38)` 會轉成新產業名稱）

## 防前視設計
- 上市天數、日均成交值過濾都是**逐日** eligibility mask，只用當日（含）以前資料。
- 族群強度在 t 日只用 t 日收盤（含）以前的資料，最早 t+1 日進場使用。
- 測試 `test_no_lookahead_truncation_invariance`：截掉未來資料後，過去每天的指標必須完全相同。
- 已知限制：產業 / 題材清單是「現在」的分類，對過去有輕微倖存者偏差。

## 族群轉強定義（參數在 config.py，先固定不優化）
`pct_above_ma20 >= 0.6` 且 `group_rank >= 0.7` 且 `pct_above_ma20` 高於 5 個交易日前；
當日有效成員 < 4 檔（或 20 日報酬窗內曾 < 4 檔）的族群不計算。
