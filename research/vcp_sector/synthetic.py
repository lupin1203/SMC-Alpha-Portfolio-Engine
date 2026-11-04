"""人造資料：給單元測試、以及沒有 FinMind / local_data 時的離線 demo 用。

模型：個股報酬 = 大盤因子 + 族群因子 + 個股雜訊。族群因子帶有輪動的 drift
（每個族群在不同時期輪流走強），因此熱力圖上應看到「強勢帶」在族群間移動。
**這不是市場資料**，只用來檢查計算與圖表流程是否正確。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import LONG_COLS


def make_synthetic_market(n_days: int = 750, seed: int = 7,
                          group_sizes: dict[str, int] | None = None,
                          start: str = "2022-01-03"):
    """回傳 ``(prices_long, stock_info, benchmark)``。"""
    rng = np.random.default_rng(seed)
    group_sizes = group_sizes or {
        "半導體業": 10, "電腦及週邊": 8, "電子零組件": 9, "航運業": 6,
        "電機機械": 7, "金融保險": 8, "生技醫療": 6, "小族群(3檔)": 3,
    }
    dates = pd.bdate_range(start, periods=n_days)
    mkt = rng.normal(0.0003, 0.010, n_days)

    rows, info = [], []
    sid = 1101
    n_groups = len(group_sizes)
    for gi, (g, size) in enumerate(group_sizes.items()):
        # 每個族群一個正弦 drift，相位錯開 → 輪動
        phase = 2 * np.pi * gi / n_groups
        drift = 0.0020 * np.sin(2 * np.pi * np.arange(n_days) / 250 + phase)
        gfac = drift + rng.normal(0, 0.008, n_days)
        for k in range(size):
            beta = rng.uniform(0.8, 1.2)
            ret = beta * mkt + gfac + rng.normal(0, 0.015, n_days)
            close = 50 * np.exp(np.cumsum(ret))
            # 第一檔晚上市：測試 min_listed_days 的逐日過濾
            listed_from = 300 if k == 0 else 0
            open_ = close * np.exp(rng.normal(0, 0.004, n_days))
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
            vol = rng.lognormal(np.log(2e6), 0.4, n_days)
            df = pd.DataFrame({"date": dates, "stock_id": str(sid), "open": open_, "high": high,
                               "low": low, "close": close, "volume": vol}).iloc[listed_from:]
            df["turnover"] = df["close"] * df["volume"]
            rows.append(df[LONG_COLS])
            info.append({"stock_id": str(sid), "stock_name": f"{g}-{k}", "industry_category": g})
            sid += 1

    prices = pd.concat(rows, ignore_index=True)
    bench = pd.Series(10000 * np.exp(np.cumsum(mkt)), index=dates, name="SYNTH_TAIEX")
    return prices, pd.DataFrame(info), bench
