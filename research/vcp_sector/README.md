# VCP + 族群連動（研究模組）

計畫：`vcp_sector_plan.md`。目前完成 **Phase 1–3（資料層、族群分群、族群強度）**，
VCP（Phase 5–6）與回測（Phase 7–8）尚未開始。

## 執行（在 repo 根目錄）

```bash
pip install pandas numpy pyyaml plotly pyarrow pytest requests
python -m research.vcp_sector.report                 # 官方產業別
python -m research.vcp_sector.report --groups theme  # groups.yaml 自訂題材
python -m research.vcp_sector.report --synthetic     # 人造資料，只檢查流程
python -m pytest research/vcp_sector/tests -q
```

輸出在 `research/vcp_sector/output/`（CSV + 熱力圖 HTML，已 gitignore）。

## 資料來源
1. 既有 `local_data/{id}.parquet`（update_data.py 產生，yfinance 還原權息）
2. `research/vcp_sector/cache/`
3. FinMind（需 `FINMIND_TOKEN`）：`TaiwanStockPriceAdj`，失敗退回 `TaiwanStockPrice` 並記 warning
- 大盤：`TAIEX`，取不到退回 `0050`
- 產業分類：FinMind `TaiwanStockInfo`，取不到退回 `stock_list.csv`

## 防前視設計
- 上市天數、日均成交值過濾都是**逐日** eligibility mask，只用當日（含）以前資料。
- 族群強度在 t 日只用 t 日收盤（含）以前的資料，最早 t+1 日進場使用。
- 測試 `test_no_lookahead_truncation_invariance`：截掉未來資料後，過去每天的指標必須完全相同。
- 已知限制：產業 / 題材清單是「現在」的分類，對過去有輕微倖存者偏差。

## 族群轉強定義（參數在 config.py，先固定不優化）
`pct_above_ma20 >= 0.6` 且 `group_rank >= 0.7` 且 `pct_above_ma20` 高於 5 個交易日前；
當日有效成員 < 4 檔（或 20 日報酬窗內曾 < 4 檔）的族群不計算。
