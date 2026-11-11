# -*- coding: utf-8 -*-
"""檢查比賽程式是否都是最新版：在每個檔案裡找只有最新版才有的內容（這支不是 esun_ 開頭，不會被打包）"""
import os

MARKERS = {
    "dplan_validator.py":     ["_schema_errors", '{"TRIM", "SELL_ALL"}'],
    "D-Plan_schema_v4_2.json": ['"4.2"'],
    "esun_daily.py":          ["step_gap_check", "window_open", "step_inst", "step_tdcc"],
    "esun_data.py":           ["fetch_finmind_twse"],
    "esun_ledger.py":         ["def reconcile"],
    "esun_orders.py":         ["def whole(q)", "CAP_TRIGGER_MARGIN", "SELL_ALL"],
    "esun_strategy.py":       ["CRASH_EXCESS = None", "def margin_score", "def bigholder_score", "FILTER_SCOPE"],
    "esun_report.py":         ["esun_version.archive()", "核心買進縮為", "大戶資料", '"schema_version": "4.2"'],
    "esun_version.py":        ["PATTERNS"],
    "esun_active_share.py":   ['"00986A", "主動台新龍頭成長", "全球"', "range(1, 12)"],
    "esun_fetch_eps.py":      ['"GrossProfit"'],
    "esun_fetch_revenue.py":  ["updated_today"],
    "esun_fetch_tdcc.py":     ["BIG_LEVELS"],
    "esun_fetch_inst.py":     ["def load"],
    "esun_notify.py":         ["utf-8-sig", "verbose"],
    "esun_strategy_doc.py":   ["投資目標與風險屬性"],
    "esun_run.bat":           ["PYTHONUTF8"],
}
ok = True
for f, marks in MARKERS.items():
    if not os.path.exists(f):
        print(f"  ✗ {f:<26} 找不到檔案"); ok = False; continue
    text = open(f, encoding="utf-8", errors="ignore").read()
    miss = [m for m in marks if m not in text]
    print(f"  {'✓' if not miss else '✗'} {f:<26} " + ("最新版" if not miss else f"舊版，缺：{miss}"))
    ok &= not miss
for f in ("esun_sector.json", "esun_etf_top10.csv", "anthropic_config.json"):
    print(f"  {'✓' if os.path.exists(f) else '✗'} {f:<26} " + ("存在" if os.path.exists(f) else "找不到"))
print("\n全部最新" if ok else "\n有檔案要更新：把 ✗ 那幾個告訴我，我重新給你")