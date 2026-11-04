"""資料 / 分群診斷：檢查上櫃股有沒有被分進族群、每個族群有幾檔可交易。

用法：python -m research.vcp_sector.diagnose
"""
from __future__ import annotations

import pandas as pd

from .config import DEFAULT_CONFIG
from .data_loader import eligibility_mask, load_prices, load_stock_info
from .sector_groups import official_groups


def main():
    cfg = DEFAULT_CONFIG
    info = load_stock_info(cfg.data, use_api=False)
    print(f"產業分類來源欄位：{info.columns.tolist()}，共 {len(info)} 檔")
    if "type" in info:
        print("\n依市場別：")
        print(info["type"].value_counts(dropna=False).to_string())
    prices = load_prices(info["stock_id"], cfg.data, use_api=False)
    have = set(prices["stock_id"].unique())
    elig = eligibility_mask(prices, cfg.data)
    last = elig.index[-1]
    elig_last = set(elig.columns[elig.loc[last].to_numpy()])
    ever = set(elig.columns[elig.any().to_numpy()])

    rows = []
    for g, ids in official_groups(info).items():
        sub = info[info["stock_id"].isin(ids)]
        row = {"group": g, "members": len(ids), "with_price": len(set(ids) & have),
               "ever_eligible": len(set(ids) & ever), f"eligible_{last:%m%d}": len(set(ids) & elig_last)}
        if "type" in sub:
            row.update(sub["type"].value_counts().to_dict())
        rows.append(row)
    df = pd.DataFrame(rows).fillna(0).sort_values("members", ascending=False)
    pd.set_option("display.width", 200)
    print("\n各族群成員數（官方產業別）：")
    print(df.to_string(index=False))
    print(f"\n有價格資料 {len(have)} 檔；曾經符合可交易條件 {len(ever)} 檔；"
          f"{last:%Y-%m-%d} 可交易 {len(elig_last)} 檔")


if __name__ == "__main__":
    main()
