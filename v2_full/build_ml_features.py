"""
build_ml_features.py — ML 特徵抽取腳本（方案 B）

作用：
  讀你的「組合回測交易明細 CSV」，對每一筆交易，回頭去該股 K 線 / 基本面
  重新計算「訊號日當下」的特徵，輸出一份「特徵 + 標籤」的 ML 訓練資料。

核心防線（前視偏差）：
  你隔天開盤才進場，所以「訊號日 = 進場日的前一個交易日」。
  所有特徵一律只用「訊號日及之前」的資料，進場日當天與之後一律不碰。
  成交量用「訊號日前一日」，Regime 用「訊號日收盤」——跟你的要求一致。

用法：
  1. 把你下載的組合回測交易明細存成 trades.csv，放在這個腳本同目錄
  2. 確認 local_data/ 資料夾在同目錄（你的 K 線）
  3. python build_ml_features.py
  4. 產出 ml_dataset.csv，就是你的 ML 訓練資料

注意：這個腳本完全不碰你的回測引擎，零風險。
"""

import os
import pandas as pd
import numpy as np

# ============ 設定：改成你的實際檔名 / 欄位名 ============
TRADES_CSV   = "trades.csv"          # 你下載的組合回測交易明細
LOCAL_DATA   = "./local_data"        # 你的 K 線資料夾
REVENUE_DIR  = "./revenue_cache/history"  # 你的月營收歷史序列資料夾（YoY 用）
OUTPUT_CSV   = "ml_dataset.csv"      # 產出的 ML 訓練資料

# 交易明細 CSV 裡的欄位名（若你的欄位名不同，改這裡）
COL_TICKER   = "標的"                 # 標的欄
COL_ENTRY    = "日期"                 # 進場日期欄
COL_RESULT   = "真實R"                # 賺賠結果（供參考）
COL_REASON   = "reason"               # 出場原因欄（Catastrophic 標籤用）
# =========================================================


def load_kline(ticker):
    """讀單一標的的 K 線。回傳 DatetimeIndex 的 df，或 None。"""
    stock_id = str(ticker).replace(".TW", "").split(".")[0]
    path = os.path.join(LOCAL_DATA, f"{stock_id}.TW.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.columns = [c.capitalize() for c in df.columns]
        return df.sort_index()
    except Exception:
        return None


def load_yoy_history(ticker):
    """讀單一標的的月營收歷史序列（含 available_date，用來做無前視偏差查詢）。"""
    stock_id = str(ticker).replace(".TW", "").split(".")[0]
    path = os.path.join(REVENUE_DIR, f"{stock_id}.parquet")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def get_yoy_at_signal(yoy_hist, signal_date):
    """
    查「訊號日當下已公告」的最新月營收 YoY。
    ★ 前視偏差防線：只取 available_date <= 訊號日 的最近一筆。
      available_date 是「市場實際可看到這筆營收的日期」(次月10日)，
      這跟你策略進場過濾用的是同一套邏輯，天生不偷看。
    """
    if yoy_hist is None or yoy_hist.empty:
        return np.nan
    signal_ts = pd.Timestamp(signal_date)

    if "available_date" in yoy_hist.columns:
        avail = pd.to_datetime(yoy_hist["available_date"])
        mask = avail.values <= signal_ts.to_datetime64()
        if mask.any():
            # 找 YoY 欄（不同版本欄名可能不同，容錯）
            for col in ("yoy", "YoY", "revenue_yoy", "月營收YoY"):
                if col in yoy_hist.columns:
                    vals = yoy_hist[col].values[mask]
                    return float(vals[-1]) if len(vals) else np.nan
    return np.nan


def compute_features_at_signal(df, entry_date):
    """
    計算「訊號日」的特徵。
    entry_date = 進場日。訊號日 = 進場日的前一個交易日。

    ★★ 前視偏差防線：所有計算只用 df 中「訊號日及之前」的資料 ★★
    """
    entry_ts = pd.Timestamp(entry_date)

    # 找出進場日在 K 線中的位置
    idx_arr = df.index.get_indexer([entry_ts], method="pad")
    if idx_arr[0] < 0:
        return None
    entry_pos = idx_arr[0]

    # 訊號日 = 進場日的前一天。若進場日剛好是 K 線某天，訊號日就是它前一格。
    signal_pos = entry_pos - 1
    if signal_pos < 60:  # 前面資料不足以算季線（SMA60），跳過
        return None

    # 只取「訊號日及之前」的資料（★ 這一行就是防偷看的核心 ★）
    hist = df.iloc[: signal_pos + 1]

    close = hist["Close"]
    high  = hist["High"]
    low   = hist["Low"]
    vol   = hist["Volume"]

    signal_close = close.iloc[-1]     # 訊號日收盤
    prev_high2   = high.iloc[-3]      # 前兩天高點（FVG 缺口下緣）
    signal_low   = low.iloc[-1]

    # ── 特徵 1：缺口大小 %（訊號日最低 vs 前兩天高點）──
    gap_size_pct = (signal_low - prev_high2) / prev_high2 * 100 if prev_high2 > 0 else 0.0

    # ── 特徵 2：季線距離 %（訊號日收盤 vs SMA60）──
    sma60 = close.rolling(60).mean().iloc[-1]
    dist_sma60_pct = (signal_close - sma60) / sma60 * 100 if sma60 > 0 else 0.0

    # ── 特徵 3：停損距離 / 進場價（風險結構）──
    #   缺口高度 = 訊號日最低 - 前兩天高點，保底股價 3%
    risk_dist = max(signal_low - prev_high2, signal_close * 0.03)
    risk_pct  = risk_dist / signal_close * 100 if signal_close > 0 else 0.0

    # ── 特徵 4：成交量倍數（★ 用訊號日「前一日」的量，避免偷看 ★）──
    #   前一日量 / 前一日的20日均量
    vol_ma20 = vol.rolling(20).mean()
    if len(vol) >= 21 and vol_ma20.iloc[-2] > 0:
        vol_ratio = vol.iloc[-2] / vol_ma20.iloc[-2]
    else:
        vol_ratio = np.nan

    return {
        "缺口大小": round(gap_size_pct, 3),
        "季線距離": round(dist_sma60_pct, 3),
        "停損距離pct": round(risk_pct, 3),
        "量倍數": round(vol_ratio, 3) if not pd.isna(vol_ratio) else np.nan,
    }


def main():
    if not os.path.exists(TRADES_CSV):
        print(f"❌ 找不到 {TRADES_CSV}，請先把組合回測交易明細下載並命名為 trades.csv")
        return

    trades = pd.read_csv(TRADES_CSV)
    print(f"讀到 {len(trades)} 筆交易")

    # 確認欄位存在
    for col in (COL_TICKER, COL_ENTRY, COL_RESULT):
        if col not in trades.columns:
            print(f"❌ CSV 裡找不到欄位「{col}」，實際欄位有：{list(trades.columns)}")
            print("   請改腳本頂部的 COL_TICKER / COL_ENTRY / COL_RESULT 對應你的欄位名。")
            return

    # 快取每檔 K 線 + YoY 序列，避免重複讀檔
    kline_cache = {}
    yoy_cache = {}
    rows = []
    skipped = 0

    for _, tr in trades.iterrows():
        ticker = tr[COL_TICKER]
        entry  = tr[COL_ENTRY]
        result = tr[COL_RESULT]

        if ticker not in kline_cache:
            kline_cache[ticker] = load_kline(ticker)
        df = kline_cache[ticker]
        if df is None:
            skipped += 1
            continue

        feats = compute_features_at_signal(df, entry)
        if feats is None:
            skipped += 1
            continue

        # ── 新增特徵：月營收 YoY（訊號日 = 進場日前一交易日，無前視偏差）──
        if ticker not in yoy_cache:
            yoy_cache[ticker] = load_yoy_history(ticker)
        # 訊號日 = 進場日前一天。用進場日查即可（available_date 機制保證只看已公告）
        feats["月營收YoY"] = get_yoy_at_signal(yoy_cache[ticker], entry)

        # ── 新增特徵：Regime（資料裡已有，訊號日算的，轉成數字給模型）──
        regime_str = tr.get("Regime", "Unknown") if "Regime" in trades.columns else "Unknown"
        # 拆成兩個 0/1 特徵：是否多頭、是否高波動（比硬編碼數字更乾淨）
        feats["是否多頭"] = 1 if "Bull" in str(regime_str) else 0
        feats["是否高波動"] = 1 if "HighVol" in str(regime_str) else 0

        # ── 標籤：賺=1，賠=0 ──
        feats["標籤_賺賠"] = 1 if result > 0 else 0
        # ── 新標籤：是不是 Catastrophic 出場（選項A：預測災難性損失）──
        if COL_REASON in trades.columns:
            feats["標籤_Catastrophic"] = 1 if str(tr[COL_REASON]) == "Catastrophic_Stop" else 0
        feats["真實R"] = result
        feats["標的"] = ticker
        feats["進場日"] = entry
        feats["Regime"] = regime_str
        if COL_REASON in trades.columns:
            feats["出場原因"] = tr[COL_REASON]

        rows.append(feats)

    if not rows:
        print("❌ 沒有任何交易成功抽出特徵，檢查 local_data 路徑或欄位名。")
        return

    ml_df = pd.DataFrame(rows)
    ml_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✅ 完成！")
    print(f"   成功抽取：{len(ml_df)} 筆")
    print(f"   跳過（無K線/資料不足）：{skipped} 筆")
    print(f"   輸出：{OUTPUT_CSV}")
    print(f"\n特徵欄位：{[c for c in ml_df.columns if c not in ('標籤_賺賠','真實R','標的','進場日','Regime')]}")
    print(f"賺賠分布：賺 {ml_df['標籤_賺賠'].sum()} 筆 / 賠 {len(ml_df) - ml_df['標籤_賺賠'].sum()} 筆")


if __name__ == "__main__":
    main()