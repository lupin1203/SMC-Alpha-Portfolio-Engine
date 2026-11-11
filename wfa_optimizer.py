# wfa_optimizer.py
"""
批次 Walk-Forward：對 stock_list.csv 裡的每一檔跑 walk_forward.run_wfo，印出盲測結果。

用法：
    python wfa_optimizer.py                 # 全部股票
    python wfa_optimizer.py 2330.TW 2317.TW # 只跑指定的幾檔

注意：這裡會用到全部歷史資料 (含 config.HOLDOUT_START 之後的考試期)，
所以預設只取研究期；要看考試期請加 --include-holdout，而且只在參數定案後看一次。
"""
import argparse
import os

import pandas as pd

from config import HOLDOUT_START
from data_provider import fetch_stock_data
from walk_forward import run_wfo

HERE = os.path.dirname(os.path.abspath(__file__))
LIST_FILE_PATH = os.path.join(HERE, "stock_list.csv")


def run_wfa_for_stock(ticker, include_holdout=False, train_years=2, test_years=1):
    df_raw = fetch_stock_data(ticker, days=3650)
    if df_raw is None or df_raw.empty:
        print(f"🚨 {ticker}: 抓不到歷史資料，跳過。")
        return None
    df_raw = df_raw.sort_index()
    if not include_holdout:
        df_raw = df_raw[df_raw.index < pd.Timestamp(HOLDOUT_START)]

    trades, logs, stats = run_wfo(df_raw, train_years=train_years, test_years=test_years)
    if stats is None:
        print(f"{ticker}: 資料不足或盲測期沒有交易。")
        return None
    print(f"{ticker}: 盲測 {stats['總交易次數']} 筆, 淨利 {stats['總淨利(R)']} R, "
          f"期望值 {stats['WFO綜合期望值']} R, 最大回撤 {stats['WFO綜合最大回撤(R)']} R")
    return {"ticker": ticker, **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", help="要跑的股票代號 (例如 2330.TW)；不給就跑 stock_list.csv 全部")
    ap.add_argument("--include-holdout", action="store_true", help="連考試期資料一起用 (參數定案後才用)")
    args = ap.parse_args()

    tickers = args.tickers
    if not tickers:
        try:
            tickers = pd.read_csv(LIST_FILE_PATH)['ticker'].dropna().astype(str).str.strip().tolist()
        except FileNotFoundError:
            raise SystemExit(f"找不到股票清單：{LIST_FILE_PATH}，請先執行 python get_data.py")
        except KeyError:
            raise SystemExit("stock_list.csv 裡找不到 'ticker' 欄位。")

    print(f"共 {len(tickers)} 檔，期間：{'全期間 (含考試期)' if args.include_holdout else f'研究期 (< {HOLDOUT_START})'}")
    rows = [r for r in (run_wfa_for_stock(t, args.include_holdout) for t in tickers) if r]
    if not rows:
        print("沒有任何一檔產生盲測交易。")
        return
    res = pd.DataFrame(rows)
    print("\n===== Walk-Forward 盲測總結 =====")
    print(f"有交易的股票: {len(res)} 檔")
    print(f"盲測總交易次數: {int(res['總交易次數'].sum())}")
    print(f"盲測總淨利: {res['總淨利(R)'].sum():.2f} R")
    print(f"盲測淨利為正的股票比例: {(res['總淨利(R)'] > 0).mean():.0%}")


if __name__ == "__main__":
    main()
