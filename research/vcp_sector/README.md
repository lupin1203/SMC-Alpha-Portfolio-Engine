# VCP + 族群連動（研究模組）

計畫：`vcp_sector_plan.md`。完成 Phase 1–8（族群強度、龍頭/落後簡化版、swing 點、VCP 偵測、
訊號 A/B/C 回測 + walk-forward），以及**半自動選股流程**（每日候選清單 + 人工判斷紀錄 + 判斷績效檢驗）。

## 日常使用（半自動）
```bash
python -m research.vcp_sector.screener      # 每天收盤後：候選清單 CSV + K 線 HTML
#   → 在 output/journal/candidates_YYYYMMDD.csv 的 decision 欄填「買」/「不買」、note 寫理由
python -m research.vcp_sector.journal       # 累積一段時間後：你挑的 vs 沒挑的 vs 全部
python -m research.vcp_sector.screener --date 2024-06-19   # 重現過去某天（練習用）
```

## 執行（在 repo 根目錄）

```bash
python -m pip install pandas numpy pyyaml plotly pyarrow pytest requests yfinance
python -m research.vcp_sector.report                 # 官方產業別
python -m research.vcp_sector.report --groups theme  # groups.yaml 自訂題材
python -m research.vcp_sector.report --synthetic     # 人造資料，只檢查流程
python -m research.vcp_sector.backtest --signal vcp # 訊號 A/C 回測（約 5–10 分鐘）
python -m research.vcp_sector.backtest --signal B   # 訊號 B 回測
python -m research.vcp_sector.vcp_review --n 30     # VCP 人工檢查頁
python -m research.vcp_sector.diagnose              # 族群分類 / 可交易股數診斷
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

## 訊號 B 回測規則（全部事先固定，在 config.py 的 TradeConfig）
- 訊號：族群 `is_strong` 由 False → True 的那天（t），選族群內龍頭（B_leader）或落後候選（B_laggard）
- 進場 t+1 開盤；停損 = t 日（含）前 10 日最低價；停損距離 < 1% 或 > 15% 不做
- 出場：盤中觸及停損（跳空則用開盤價）；收盤 < 進場後最高收盤 × 0.9、收盤由上往下跌破 MA20、
  持有 120 日 → 隔日開盤出
- 成本：手續費 0.1425% ×2、證交稅 0.3%、滑價 0.1% ×2
- 投組：每筆風險 1%、最多 10 檔、單檔上限 20%、可零股
- 比較：同日同筆數的隨機進場 baseline（20 組）、分年度、分 regime、族群門檻 3×3 敏感度、
  walk-forward（IS 3 年選參數 → OOS 1 年，IS 只計入在 OOS 開始前已出場的交易）

## VCP 規則（vcp_detector.py；參數在 config.py 的 VCPConfig）
趨勢樣板 → 近 120 日最高 swing High 起算，2–4 段收縮逐段變淺（≤ 前段 × 0.8，首段 ≤ 35%、末段 ≤ 10%）
→ 末段均量 < 整理前 50 日均量 × 0.7 → 突破 pivot 且量 > 前 50 日均量 × 1.5。
人工檢查 30 例後加入（依看圖判斷，非依回測結果）：整理期 ≥ 20 交易日、pivot 距整理頂 ≤ 10%、
隔日開盤 > pivot × 1.05 不追。

## 目前的回測結論（2016–2026，固定參數）
- 訊號 B（族群剛轉強就買）：負期望值，與隨機進場無異，參數怎麼調都一樣。
- VCP（加入整理期/pivot 規則後）：A 約 +0.06R，但 10 年僅約 100 筆、t 值 < 1，無法與運氣區分。
- 族群過濾（C）沒有讓 VCP 變好（樣本僅十幾筆）。
- 因此改為半自動：程式篩、人工判斷，並用 journal 檢驗人工判斷是否真的加分。
