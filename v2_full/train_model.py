"""
train_model.py — 選項A：預測 Catastrophic 風險

目的：
  不預測「會不會賺」（已證明預測不了），改預測「這筆會不會演變成災難性損失
  (Catastrophic_Stop)」。若能事前識別高風險訊號，就能降低部位或跳過，
  直接減少你最大的失血點（Catastrophic 總計 -548R）。

★ 這次判讀重點是「召回率(recall)」不是準確率：
  目的是「避開災難」，所以我們要模型盡量抓到真正的 Catastrophic
  （寧可錯殺一些正常單，也要盡量標記出真災難）。

★ 假設：風險比報酬更容易事前預測（下跌動力學會留事前痕跡）。
  這個腳本就是在驗證這個假設成不成立。
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

DATASET = "ml_dataset.csv"
FEATURES = ["缺口大小", "季線距離", "停損距離pct", "量倍數",
            "月營收YoY", "是否多頭", "是否高波動"]
LABEL = "標籤_Catastrophic"


def main():
    df = pd.read_csv(DATASET)
    df = df.sort_values("進場日").reset_index(drop=True)

    tech = ["缺口大小", "季線距離", "停損距離pct", "量倍數"]
    df = df.dropna(subset=tech).reset_index(drop=True)
    if "月營收YoY" in df.columns and df["月營收YoY"].isna().any():
        df["月營收YoY"] = df["月營收YoY"].fillna(df["月營收YoY"].median())

    if LABEL not in df.columns:
        print(f"[X] 找不到欄位 {LABEL}，請先重跑 build_ml_features.py 產生新的 ml_dataset.csv")
        return

    X = df[FEATURES]
    y = df[LABEL]

    n_cata = int(y.sum())
    print(f"資料 {len(df)} 筆，其中 Catastrophic {n_cata} 筆（{n_cata/len(df):.1%}）")

    split = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"訓練集 {len(X_train)}（早期）｜測試集 {len(X_test)}（近期）")

    # 基準線：全猜「不是Catastrophic」
    baseline_acc = (y_test == 0).mean()
    print("\n" + "=" * 55)
    print(f"【基準線】測試集全猜『非災難』準確率 = {baseline_acc:.1%}")
    print("  但注意：這次我們更在乎『召回率』——能抓到多少真災難")
    print("=" * 55)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=20,
        class_weight="balanced", random_state=42,
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_pred = model.predict(X_test)
    test_acc  = accuracy_score(y_test, test_pred)

    gap = train_acc - test_acc
    tag = " [!]過擬合" if gap > 0.10 else " [ok]"
    print(f"\n訓練集準確率：{train_acc:.1%}｜測試集準確率：{test_acc:.1%}  (差 {gap:.1%}{tag})")

    cm = confusion_matrix(y_test, test_pred)
    print("\n【混淆矩陣】")
    print("               預測非災難  預測災難")
    print(f"  實際非災難    {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"  實際災難      {cm[1,0]:6d}   {cm[1,1]:6d}")

    actual_cata = cm[1,0] + cm[1,1]
    caught_cata = cm[1,1]
    pred_cata   = cm[0,1] + cm[1,1]

    print("\n" + "─" * 55)
    if actual_cata > 0:
        recall = caught_cata / actual_cata
        print(f"★召回率：測試集實際 {actual_cata} 筆災難，模型抓到 {caught_cata} 筆 = {recall:.1%}")
        print(f"  （這是重點！越高代表越能事前標記出真災難去避開）")
    if pred_cata > 0:
        precision = caught_cata / pred_cata
        print(f" 精準率：模型喊災難 {pred_cata} 次，對 {caught_cata} 次 = {precision:.1%}")
        print(f"  （模型喊災難時的可信度。太低代表誤殺太多正常單）")

    # 對照：隨機亂猜的召回率會是多少？（等於災難佔比）
    base_rate = actual_cata / len(y_test)
    print(f"\n 對照：若隨機標記，精準率約 = 災難佔比 {base_rate:.1%}")
    if pred_cata > 0 and precision > base_rate + 0.03:
        print(f"  → [V] 模型精準率 {precision:.1%} > 隨機 {base_rate:.1%}，模型有抓到一些真訊號")
    else:
        print(f"  → [X] 模型精準率沒明顯贏過隨機，預測力有限")
    print("─" * 55)

    print("\n【特徵重要性】哪個線索最能預示災難：")
    imps = sorted(zip(FEATURES, model.feature_importances_),
                  key=lambda x: x[1], reverse=True)
    for name, imp in imps:
        bar = "#" * int(imp * 50)
        print(f"  {name:12s} {imp:.3f} {bar}")

    print("\n" + "=" * 55)
    print("判讀：召回率高 + 精準率贏過隨機 = ML 能幫你事前避開部分災難。")
    print("即使只能避開一部分，對你最大失血點(Catastrophic)都是實質改善。")
    print("=" * 55)


if __name__ == "__main__":
    main()