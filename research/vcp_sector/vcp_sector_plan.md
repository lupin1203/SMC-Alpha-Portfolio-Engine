# VCP + 族群連動 實作大綱

> 目標：在現有 SMC- 專案（Python、FinMind 資料）中新增兩個模組，**族群強度**與 **VCP 型態偵測**，最後組合成「族群轉強 + VCP 突破」訊號並回測。
> 原則：先做族群強度（簡單、穩定），再做 VCP；參數少、可解釋；所有回測必須做樣本外驗證，避免過擬合。

---

## 0. 專案結構

```
SMC-/
└── research/vcp_sector/
    ├── config.py            # 參數集中管理
    ├── data_loader.py       # 讀 FinMind 日K、產業分類（沿用現有資料快取）
    ├── sector_groups.py     # 族群分群
    ├── sector_strength.py   # 族群強度指標
    ├── lead_lag.py          # 龍頭 / 落後股判斷
    ├── swing_points.py      # 波段高低點偵測
    ├── vcp_detector.py      # VCP 型態辨識
    ├── signals.py           # 組合訊號
    ├── backtest.py          # 回測 + walk-forward
    ├── report.py            # 統計輸出 / Streamlit 頁面
    └── tests/               # 單元測試（用人工造的價格序列）
```

---

## Phase 1：資料層 `data_loader.py`

- 讀取個股日K：open / high / low / close / volume（還原權息價）
- 讀取產業分類（FinMind `TaiwanStockInfo` 的 industry_category）
- 統一輸出格式：`DataFrame[date, stock_id, open, high, low, close, volume]`
- 過濾：剔除 ETF、權證、上市未滿 250 日、日均成交值過低的個股（門檻放 config）
- 優先讀現有快取，沒有才打 API

---

## Phase 2：族群分群 `sector_groups.py`

提供三種分群方式，輸出 `dict[group_name, list[stock_id]]`：

1. **官方產業別**（baseline）：直接用 industry_category
2. **自訂題材清單**：讀 `groups.yaml`（例：AI伺服器、散熱、CoWoS、重電、矽光子），手動維護
3. **相關係數分群**（進階，可最後做）：
   - 用過去 120 日日報酬算相關矩陣
   - 距離 = 1 − corr，做 hierarchical clustering
   - 每月重算一次，避免前視偏差（只用當時以前的資料）

---

## Phase 3：族群強度 `sector_strength.py`

每個族群、每個交易日計算：

| 指標 | 定義 |
|---|---|
| `pct_above_ma20` | 成員中收盤 > MA20 的比例 |
| `pct_above_ma50` | 成員中收盤 > MA50 的比例 |
| `pct_new_high_20` | 成員中創 20 日新高的比例 |
| `group_ret_20` | 族群等權指數 20 日報酬 |
| `group_rs` | 族群 20 日報酬 − 大盤（TAIEX）20 日報酬 |
| `group_rank` | `group_rs` 在所有族群中的百分位排名 |

- **族群轉強定義**（先用簡單版本，參數放 config）：
  `pct_above_ma20 >= 0.6` 且 `group_rank >= 0.7` 且 `pct_above_ma20` 比 5 日前上升
- 族群成員數 < 4 的不計算（太少沒有統計意義）
- 輸出：`DataFrame[date, group, 各指標, is_strong]`

---

## Phase 4：龍頭 / 落後 `lead_lag.py`

- **龍頭**：族群內近 20 日報酬最高、且市值或成交值排前段的 1–2 檔
- **落後候選**：同族群、族群 `is_strong == True`、但個股 20 日報酬 < 族群中位數，且仍在 MA50 之上（避免撿到弱勢股）
- （進階）lead-lag：計算龍頭與成員的 cross-correlation（lag 1–5 日），找出穩定跟漲的成員
- 輸出：每日 `leaders`、`laggards` 清單

---

## Phase 5：波段高低點 `swing_points.py`

VCP 的核心難點，獨立成模組並寫測試。

- 方法：**ZigZag 百分比法**，價格反轉超過 `min_swing_pct`（例如 4%）才確認一個新的 swing high/low
- 只能用當下以前的資料確認轉折點（**不可前視**）——轉折點要等反轉幅度成立後才算確認，回測時用「確認日」而非「轉折發生日」
- 輸出：`list[(date, price, 'H' | 'L', confirm_date)]`

---

## Phase 6：VCP 辨識 `vcp_detector.py`

對每檔股票每日判斷是否符合 VCP，條件全部參數化：

1. **前置趨勢（Trend Template 簡化版）**
   - close > MA50 > MA150 > MA200
   - MA200 比 20 日前高
   - close 距 52 週高點 ≤ 25%，距 52 週低點 ≥ 30%
2. **收縮結構**
   - 從近 `lookback`（例 60–120 日）內找最高點作為整理起點
   - 之後取 2–4 段回檔（swing high → swing low）
   - 每段回檔幅度 < 前一段（允許容忍度，如 ≤ 前段 × 0.8 視為收縮）
   - 第一段回檔 ≤ 35%，最後一段 ≤ 10%
3. **量縮**
   - 最後一段收縮期間的均量 < 整理前 50 日均量 × `vol_dry_ratio`（例 0.7）
4. **Pivot**
   - pivot = 最後一段收縮的 swing high
5. **突破觸發**
   - 收盤 > pivot，且當日量 > 50 日均量 × `breakout_vol_mult`（例 1.5）
- 輸出：`DataFrame[date, stock_id, is_vcp_setup, pivot, stop_price, n_contractions, contraction_depths]`

---

## Phase 7：組合訊號 `signals.py`

三組訊號分開產生，回測時可以比較：

- **A. 純 VCP 突破**
- **B. 純族群強度**：族群剛轉強時，買進族群內強勢股或落後候選
- **C. VCP 突破 + 所屬族群 `is_strong`**（主要假設：C 優於 A）

進出場規則（先固定，不要優化）：
- 進場：訊號日隔日開盤
- 停損：VCP 最後一段收縮低點（`stop_price`）
- 出場：移動停利（沿用 SMC 的 trailing stop 邏輯）或跌破 MA20
- 部位大小：固定風險（每筆風險 = 帳戶 1%，用 R 計算）

---

## Phase 8：回測 `backtest.py`

- 輸出指標：交易次數、勝率、平均 R、期望值（R）、最大回撤、年化報酬
- 分年度統計（看是否只在 2020、2023 這類趨勢年賺錢）
- 分 regime 統計（沿用 SMC 的四格 regime：Bull/Bear × High/Low Vol）
- **Walk-forward**：in-sample 3 年 / out-of-sample 1 年，滾動
- **比較基準**：A vs C 的期望值差異、隨機進場同期間的 baseline
- 參數敏感度：對 `min_swing_pct`、`vol_dry_ratio`、族群門檻做小範圍網格，看結果是否穩定（穩定比最高分重要）

---

## Phase 9：輸出 `report.py`

- CSV：每日訊號清單、交易明細
- Streamlit 頁面：
  - 族群強度熱力圖（族群 × 日期）
  - 今日轉強族群 + 龍頭 / 落後清單
  - 今日 VCP setup 清單（含 pivot、停損、收縮次數）
  - 個股 K 線標註 swing 點與 pivot（用於人工檢查辨識是否正確）

---

## 測試重點 `tests/`

- `swing_points`：用人造鋸齒價格，確認轉折點與確認日正確、沒有前視
- `vcp_detector`：人造一個標準 VCP（25% → 12% → 6% + 量縮）要被抓到；回檔遞增的不能被抓到
- `sector_strength`：手算小樣本比例對得上
- 回測：檢查所有訊號只用到訊號日（含）以前的資料

---

## 建議開發順序

1. Phase 1 → 2（官方產業 + 自訂 yaml）→ 3 → 先做訊號 B 的簡單回測
2. Phase 5 → 6 → 用 Streamlit K 線人工看 20–30 個案例確認辨識品質
3. Phase 7 → 8 比較 A / B / C
4. 最後才做相關係數分群與 lead-lag
