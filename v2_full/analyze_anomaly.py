"""
analyze_anomaly.py — 深挖「季線距離 15~20%」這個風險報酬異常

這格 143 筆：災難率 37.8%（高），但平均只有 0.22R（低），
脫離了「風險越高報酬越高」的規律。這腳本回答三個問題：
  1. 這是真的異常，還是 143 筆樣本太少的雜訊？
  2. 若是真的，這格的交易集中在哪些板塊/年份？
  3. 它跟隔壁兩組（10~15%、20%+）到底差在哪？

用法：python analyze_anomaly.py （讀 ml_dataset.csv）
"""

import pandas as pd
import numpy as np

DATASET = "ml_dataset.csv"


def main():
    df = pd.read_csv(DATASET)
    df = df.dropna(subset=["季線距離"]).reset_index(drop=True)

    # 三個相鄰組
    g_mid  = df[(df["季線距離"] > 10) & (df["季線距離"] <= 15)].copy()  # 10~15
    g_anom = df[(df["季線距離"] > 15) & (df["季線距離"] <= 20)].copy()  # 15~20 異常
    g_high = df[(df["季線距離"] > 20)].copy()                          # 20+

    print("=" * 60)
    print("問題1：這個異常是真的，還是雜訊？")
    print("=" * 60)
    r = g_anom["真實R"]
    print(f"15~20% 這格共 {len(r)} 筆")
    print(f"  平均R：{r.mean():.2f}")
    print(f"  中位數R：{r.median():.2f}   ← 若中位數跟平均差很多，代表被極端值拉歪")
    print(f"  最大R：{r.max():.2f}｜最小R：{r.min():.2f}")
    print(f"  賺的筆數：{(r>0).sum()}／賠的筆數：{(r<0).sum()}")
    # 看有沒有大賺的（移動停利那種）
    big_win = (r > 5).sum()
    print(f"  R>5 的大賺：{big_win} 筆（{big_win/len(r):.1%}）")
    print(f"  → 若大賺特別少，就能解釋為什麼這格報酬低：抓不到金雞母")

    # 對照：另兩組的大賺比例
    print(f"\n  對照大賺(R>5)比例：")
    print(f"    10~15%組：{(g_mid['真實R']>5).mean():.1%}")
    print(f"    15~20%組：{(g_anom['真實R']>5).mean():.1%}  ← 異常組")
    print(f"    20%+ 組：{(g_high['真實R']>5).mean():.1%}")

    print("\n" + "=" * 60)
    print("問題2：這格的交易集中在哪些板塊？（前6）")
    print("=" * 60)
    print(g_anom["板塊"].value_counts().head(6) if "板塊" in g_anom.columns else "（無板塊欄）")

    print("\n" + "=" * 60)
    print("問題3：三組並排比較")
    print("=" * 60)
    print(f"{'指標':<16}{'10~15%':>12}{'15~20%':>12}{'20%+':>12}")
    print("-" * 52)
    for name, col, fmt in [
        ("筆數", None, None),
        ("平均R", "真實R", "{:.2f}"),
        ("中位數R", "真實R", "{:.2f}"),
        ("災難率", "標籤_Catastrophic", "{:.1%}"),
        ("大賺率(R>5)", None, None),
        ("勝率", None, None),
    ]:
        if name == "筆數":
            print(f"{name:<16}{len(g_mid):>12}{len(g_anom):>12}{len(g_high):>12}")
        elif name == "平均R":
            print(f"{name:<16}{g_mid['真實R'].mean():>12.2f}{g_anom['真實R'].mean():>12.2f}{g_high['真實R'].mean():>12.2f}")
        elif name == "中位數R":
            print(f"{name:<16}{g_mid['真實R'].median():>12.2f}{g_anom['真實R'].median():>12.2f}{g_high['真實R'].median():>12.2f}")
        elif name == "災難率":
            print(f"{name:<16}{g_mid['標籤_Catastrophic'].mean():>11.1%}{g_anom['標籤_Catastrophic'].mean():>12.1%}{g_high['標籤_Catastrophic'].mean():>12.1%}")
        elif name == "大賺率(R>5)":
            print(f"{name:<16}{(g_mid['真實R']>5).mean():>11.1%}{(g_anom['真實R']>5).mean():>12.1%}{(g_high['真實R']>5).mean():>12.1%}")
        elif name == "勝率":
            print(f"{name:<16}{(g_mid['真實R']>0).mean():>11.1%}{(g_anom['真實R']>0).mean():>12.1%}{(g_high['真實R']>0).mean():>12.1%}")

    print("\n" + "=" * 60)
    print("判讀：")
    print("  · 若異常組『中位數R』跟別組差不多，只是平均被少數極端賠拉低 → 偏雜訊")
    print("  · 若異常組『大賺率』明顯低於別組 → 真異常：這格抓不到金雞母")
    print("    （高風險卻沒有高報酬補償 = 風險報酬最差的甜蜜點反面）")
    print("=" * 60)


if __name__ == "__main__":
    main()