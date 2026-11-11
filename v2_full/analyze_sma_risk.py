"""
analyze_sma_risk.py — 深挖「季線距離 vs Catastrophic 機率」

不訓練模型，純資料分析：把交易依「進場時離季線多遠」分組，
算每組的 Catastrophic 機率，看是不是「離季線越遠、越容易踩雷」。

這是最誠實的分析——直接看資料本身，不經過任何模型。
若趨勢清楚，你就得到一個可直接用的策略洞察。

用法：python analyze_sma_risk.py （讀 ml_dataset.csv）
"""

import pandas as pd
import numpy as np

DATASET = "ml_dataset.csv"


def main():
    df = pd.read_csv(DATASET)

    if "標籤_Catastrophic" not in df.columns or "季線距離" not in df.columns:
        print("[X] 缺欄位，請先重跑 build_ml_features.py")
        return

    df = df.dropna(subset=["季線距離"]).reset_index(drop=True)
    total_cata_rate = df["標籤_Catastrophic"].mean()
    print(f"全體 {len(df)} 筆，整體 Catastrophic 機率 = {total_cata_rate:.1%}")
    print("=" * 60)

    # 依季線距離分組（自訂區間）
    bins   = [-100, 0, 5, 10, 15, 20, 100]
    labels = ["低於季線", "0~5%", "5~10%", "10~15%", "15~20%", "20%以上"]
    df["季線距離組"] = pd.cut(df["季線距離"], bins=bins, labels=labels)

    print(f"\n{'季線距離':<12}{'交易數':>8}{'災難數':>8}{'災難機率':>10}   趨勢圖")
    print("-" * 60)
    for grp in labels:
        sub = df[df["季線距離組"] == grp]
        if len(sub) == 0:
            continue
        n = len(sub)
        cata = int(sub["標籤_Catastrophic"].sum())
        rate = cata / n
        bar = "#" * int(rate * 50)
        # 標記高於/低於整體平均
        flag = "  ⬆高風險" if rate > total_cata_rate + 0.05 else ("  ⬇低風險" if rate < total_cata_rate - 0.05 else "")
        print(f"{grp:<12}{n:>8}{cata:>8}{rate:>9.1%}   {bar}{flag}")

    print("-" * 60)
    print(f"{'（整體平均）':<12}{len(df):>8}{int(df['標籤_Catastrophic'].sum()):>8}{total_cata_rate:>9.1%}")

    # 額外：也看看這些組的平均報酬（R），看高風險組是不是也低報酬
    print("\n【對照】各組的平均真實R（看高風險組是不是也賺得少）：")
    print("-" * 60)
    for grp in labels:
        sub = df[df["季線距離組"] == grp]
        if len(sub) == 0:
            continue
        avg_r = sub["真實R"].mean()
        print(f"{grp:<12}平均 {avg_r:>6.2f}R   （{len(sub)} 筆）")

    print("\n" + "=" * 60)
    print("判讀：若『災難機率』隨季線距離拉遠而明顯上升，")
    print("      代表『追高進場』確實容易踩雷，這就是可用的策略洞察。")
    print("      再對照平均R——若高風險組報酬也差，那更該避開。")
    print("=" * 60)


if __name__ == "__main__":
    main()