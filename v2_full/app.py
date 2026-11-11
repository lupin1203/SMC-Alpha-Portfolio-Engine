import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import logging
import requests
import datetime
import os          
import json
from dataclasses import dataclass
from live_execution import send_telegram_message
from utils import safe_divide

# ==========================================
# 宏觀板塊動態對接
# ==========================================
from dynamic_sector import get_sector_map
# 拔除舊版死代碼：from sector_radar import SectorRadar

st.set_page_config(page_title="SMC 量化拷問終端 V20.0", layout="wide") # 升級為 V20.0

# 初始化 Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_provider import load_data, fetch_stock_data
from strategy import apply_smc_logic
from backtest_engine import run_backtest_core
from database import init_db, save_backtest_record, load_history, save_backtest_records_batch
from revenue_fetcher import screen_sector_by_revenue
from monte_carlo import run_monte_carlo
from portfolio_engine import run_portfolio_backtest

@dataclass
class PortfolioConfig:
    """投資組合回測參數包裝器，避免多位置參數傳遞錯誤"""
    tickers: list
    years: int
    bt_mode: str
    bt_val: float
    gap_pct: float
    use_sma: bool
    use_vol_filter: bool
    use_atr_filter: bool
    use_bos_filter: bool
    slippage_pct: float
    fee_disc: float
    min_vol: int
    max_concurrent: int
    max_per_sector: int
    sector_map: dict
    risk_pct: float
    initial_capital: float
    account_dd_limit: float
    max_position_pct: float = 0.15
    max_corr: float = 0.75
    allowed_regimes: list = None      # None = 全部環境
    min_holding_days: int = 0         # 0 = 不過濾
    regime_benchmark: str = "0050.TW" # Regime 基準
    vol_threshold: float = 25.0       # 高波動門檻 (%)
    entry_min_yoy: float = None       # 進場要求月營收 YoY ≥ 此值
    entry_require_gm: bool = False     # 進場要求毛利率擴張
    entry_require_ocf: bool = False    # 進場要求營業現金流為正（地雷股防禦）
    enable_sector_rotation: bool = False  # 動態板塊輪動（每月選最強 N 板塊）
    sector_top_n: int = 3
    full_sector_map: dict = None       # 全板塊地圖（輪動評分用）

@st.cache_data(ttl=3600)
def get_cached_portfolio_results(cfg: PortfolioConfig):
    return run_portfolio_backtest(
        tickers=cfg.tickers,
        years=cfg.years,
        bt_mode=cfg.bt_mode,
        bt_val=cfg.bt_val,
        gap_pct=cfg.gap_pct,
        use_sma=cfg.use_sma,
        use_vol_filter=cfg.use_vol_filter,
        use_atr_filter=cfg.use_atr_filter,
        use_bos_filter=cfg.use_bos_filter,
        slippage_pct=cfg.slippage_pct,
        fee_disc=cfg.fee_disc,
        min_vol=cfg.min_vol,
        max_concurrent=cfg.max_concurrent,
        max_per_sector=cfg.max_per_sector,
        sector_map=cfg.sector_map,
        risk_pct=cfg.risk_pct,
        initial_capital=cfg.initial_capital,
        account_dd_limit=cfg.account_dd_limit,
        max_position_pct=cfg.max_position_pct,
        max_corr=cfg.max_corr,
        allowed_regimes=cfg.allowed_regimes,
        min_holding_days=cfg.min_holding_days,
        regime_benchmark=cfg.regime_benchmark,
        vol_threshold=cfg.vol_threshold,
        entry_min_yoy=cfg.entry_min_yoy,
        entry_require_gm=cfg.entry_require_gm,
        entry_require_ocf=cfg.entry_require_ocf,
        enable_sector_rotation=cfg.enable_sector_rotation,
        sector_top_n=cfg.sector_top_n,
        full_sector_map=cfg.full_sector_map,
    )

# 初始化資料庫與靜態數據
init_db()
INDUSTRY_DICT, STOCK_NAMES, SEARCH_LIST = load_data()

# ==========================================
# 雷達 UI 設定區塊 (側邊欄)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("宏觀板塊動態對接")
force_update_map = st.sidebar.button("強制更新 FinMind 產業地圖")

with st.spinner("載入動態產業地圖中..."):
    dynamic_map = get_sector_map(force_update=force_update_map)
    SECTOR_MAP = dynamic_map if dynamic_map else INDUSTRY_DICT
    st.sidebar.success(f"已載入 {len(SECTOR_MAP)} 個產業分類")
# ==========================================

st.title("SMC 台股選股 V20.0 ")
st.caption("Tier 5 機構級驗證：動態部位縮減、移動停利解鎖、蒙地卡羅壓力測試")

# ==========================================
# 資料新鮮度時間戳（A: 現在時間 + B: 資料最後更新）
# ==========================================
def _get_data_freshness():
    """回傳 (現在時間字串, 資料最後更新日字串, 是否過期)"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    data_date_str = "無資料"
    is_stale = True
    try:
        _p = "./local_data/0050.TW.parquet"   # 用0050當代表(一定有)
        if os.path.exists(_p):
            _d = pd.read_parquet(_p)
            if not _d.empty:
                _last = pd.to_datetime(_d.index).max()
                data_date_str = _last.strftime("%Y-%m-%d")
                _gap = (datetime.date.today() - _last.date()).days
                is_stale = _gap > 4   # 超過4天沒更新就示警(容忍週末+假日)
    except Exception:
        pass
    return now_str, data_date_str, is_stale

_now_str, _data_date, _is_stale = _get_data_freshness()
_ts_c1, _ts_c2 = st.columns(2)
_ts_c1.metric("現在時間", _now_str)
if _is_stale:
    _ts_c2.metric("資料最後更新", _data_date, delta="已過期！請跑 update_data.py", delta_color="inverse")
    st.error(f"⚠️ K線資料最後更新是 {_data_date}，可能已過期。請執行 python update_data.py 更新後再掃描。")
else:
    _ts_c2.metric("資料最後更新", _data_date, delta="資料新鮮", delta_color="normal")

# 分頁宣告
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "掃描雷達", "個股回測", "投資組合回測 (Portfolio)",
    "防禦與衰減檢定", "實盤對帳", "參數變更日誌","歷史資料庫"
])

# ==========================================
#  分頁 1：掃描雷達
# ==========================================
with tab1:
    st.header("今日搜尋進場個股")
    st.caption("同步回測邏輯：導入機構級部位派單與 SMC 動態濾網")

    # ── 獨立外資買賣超查詢（隨時可查，不受當日有無訊號影響）──
    with st.expander("🔍 外資買賣超查詢（下單前雙重確認用，不影響策略）", expanded=False):
        st.caption("策略掃出訊號後，可在這裡查該股外資籌碼面當作下單前確認。僅供人工參考，不改變系統訊號。")
        _chip_input = st.text_input("輸入股票代碼（可多檔，逗號分隔，例 2330, 2454）", key="chip_query_input")
        if st.button("查詢外資買賣超", key="chip_query_btn"):
            if not _chip_input.strip():
                st.warning("請先輸入至少一個股票代碼。")
            else:
                from revenue_fetcher import fetch_institutional_net
                _codes = [c.strip().replace(".TW", "") for c in _chip_input.split(",") if c.strip()]
                _chip_rows = []
                _prog = st.progress(0.0)
                for _i, _code in enumerate(_codes):
                    _prog.progress((_i + 1) / len(_codes))
                    _nm = STOCK_NAMES.get(_code, STOCK_NAMES.get(f"{_code}.TW", "—"))
                    try:
                        _inst = fetch_institutional_net(_code)
                        if _inst.get("data_ok"):
                            _chip_rows.append({"股票代碼": _code, "股票名稱": _nm,
                                "外資買賣超(張)": _inst.get("foreign_net"),
                                "外資連買天數": _inst.get("foreign_streak"),
                                "近5日三大法人(張)": _inst.get("total_net_5d")})
                        else:
                            _chip_rows.append({"股票代碼": _code, "股票名稱": _nm,
                                "外資買賣超(張)": "無資料", "外資連買天數": "—", "近5日三大法人(張)": "—"})
                    except Exception:
                        _chip_rows.append({"股票代碼": _code, "股票名稱": _nm,
                            "外資買賣超(張)": "查詢失敗", "外資連買天數": "—", "近5日三大法人(張)": "—"})
                _prog.empty()
                if _chip_rows:
                    _chip_df = pd.DataFrame(_chip_rows)
                    _chip_df["_sk"] = pd.to_numeric(_chip_df["外資買賣超(張)"], errors="coerce")
                    _chip_df = _chip_df.sort_values("_sk", ascending=False, na_position="last").drop(columns=["_sk"])
                    st.dataframe(_chip_df, use_container_width=True, hide_index=True)
                    st.caption("已依外資買賣超排序（正=買超）。連買+你的FVG訊號=雙重確認。但連買也可能是出貨末端，僅供參考。")

    if 'radar_dfs' not in st.session_state:
        st.session_state['radar_dfs'] = {}
    
    INDUSTRY_DICT_REVERSE = {ticker: sec for sec, tickers in SECTOR_MAP.items() for ticker in tickers}
    
    st.markdown("### 實盤部位與資金設定")
    col_risk1, col_risk2, col_risk3 = st.columns(3)
    with col_risk1:
        live_equity = st.number_input("實盤帳戶總淨值 (NTD)", value=100000, step=5000)
        live_risk_pct = st.number_input("今日單筆風險 (%)", value=0.5, step=0.1) / 100.0
    with col_risk2:
        min_momentum = st.number_input("最低動能要求 (倍)", value=2.0, step=0.5)
        min_rr = st.number_input("最低預期賠率 (R)", value=2.0, step=0.5)
    with col_risk3:
        max_capital_per_trade = st.number_input("單筆資金佔比上限 (%)", value=20.0, step=5.0) / 100.0
        trailing_stop_pct = st.number_input("移動停利距離 (%)", value=10.0, step=1.0,
            help="掃描結果的獲利目標 = 進場價 × (1 + 此值%)，與回測設定保持一致") / 100.0

    # ── Regime 過濾設定 ──
    st.markdown("### 市場環境設定")

    # 基準指數選擇 + 波動率門檻
    bench_col1, bench_col2 = st.columns(2)
    with bench_col1:
        regime_benchmark = st.selectbox(
            "Regime 判斷基準",
            ["0050 權值股", "加權指數 (全市場)", "櫃買指數 (中小型股)"],
            index=0,
            help="0050 只含 50 檔權值股，若你主要交易中小型成長股，"
                 "用櫃買指數判斷環境會更貼近實際。大盤與個股脫鉤時尤其重要。",
        )
    with bench_col2:
        vol_threshold = st.slider(
            "高波動門檻 (年化波動率 %)",
            min_value=15, max_value=50, value=25, step=5,
            help="年化波動率超過此值判定為 HighVol。台股近年波動較大，"
                 "可視情況放寬。調整後請回 Tab3 重新回測確認期望值仍為正。",
        )

    # 基準指數對應
    #   0050 走本地快取（yfinance），加權/櫃買走 FinMind（台灣本土資料，較準）
    _bench_map = {
        "0050 權值股":        ("yf", "0050.TW"),
        "加權指數 (全市場)":   ("finmind", "TAIEX"),
        "櫃買指數 (中小型股)": ("finmind", "TPEx"),
    }
    bench_source, bench_id = _bench_map.get(regime_benchmark, ("yf", "0050.TW"))

    def _get_benchmark_data(days=250):
        if bench_source == "finmind":
            from revenue_fetcher import fetch_taiwan_index
            return fetch_taiwan_index(bench_id, days=days)
        return fetch_stock_data(bench_id, days=days)

    reg_col1, reg_col2 = st.columns(2)
    with reg_col1:
        # 計算當前 Regime（用選定的基準和門檻）
        _mbench = _get_benchmark_data(days=250)
        if _mbench is not None and not _mbench.empty and len(_mbench) > 200:
            _close = _mbench['Close'].iloc[-1]
            _sma200 = _mbench['Close'].rolling(200).mean().iloc[-1]
            _vol20  = _mbench['Close'].pct_change().rolling(20).std().iloc[-1] * (252**0.5) * 100
            _trend  = "Bull" if _close > _sma200 else "Bear"
            _volreg = "HighVol" if _vol20 > vol_threshold else "LowVol"
            current_regime_now = f"{_trend}_{_volreg}"
        else:
            current_regime_now = "Unknown"
            _vol20 = 0

        st.metric("當前市場環境", current_regime_now,
                  help=f"基準：{regime_benchmark}。Bull/Bear 依是否在 200MA 上方；"
                       f"HighVol 依近 20 日年化波動率是否 > {vol_threshold}%")
        if current_regime_now != "Unknown":
            st.caption(f"目前 {regime_benchmark} 年化波動率：{_vol20:.1f}%（門檻 {vol_threshold}%）")

    with reg_col2:
        scan_allowed_regimes = st.multiselect(
            "允許發出訊號的 Regime",
            ["Bull_LowVol", "Bull_HighVol", "Bear_LowVol", "Bear_HighVol"],
            default=["Bull_LowVol", "Bear_HighVol"],
            help="只在正期望值的市場環境發出訊號，其他環境自動靜音",
        )

    st.markdown("### 掃描條件與 SMC 濾網設定")
    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        r_gap = st.number_input("最小缺口(%)", value=0.3, step=0.1, key="r_g")
    with c2:
        r_sma = st.checkbox("啟用季線濾網", value=True, key="r_s")
        r_vol = st.checkbox("要求起漲爆量", value=False, key="r_v")
    with c3:
        r_atr = st.checkbox("強勢實體位移", value=False, key="r_a")
        r_bos = st.checkbox("突破近期新高", value=False, key="r_b")
    with c4:
        sector_keys = list(SECTOR_MAP.keys())
        r_sectors = st.multiselect("自選掃描板塊 (預設全市場)", sector_keys, default=sector_keys)
        use_auto_radar = st.checkbox("啟用 AI 宏觀雷達 (自動精煉最強資金板塊)", value=True)
        top_n_sectors = st.number_input("AI 雷達精煉板塊數量", min_value=1, max_value=10, value=3, step=1) if use_auto_radar else 3

    # ==========================================
    # 月營收動能篩選設定（新增）
    # ==========================================
    if use_auto_radar:
        st.markdown("### 月營收動能篩選條件（第一道鎖）")
        rev_col1, rev_col2, rev_col3 = st.columns(3)
        with rev_col1:
            use_revenue_filter = st.checkbox("啟用月營收 YoY 篩選", value=True,
                help="從 FinMind 抓取月營收，只保留 YoY 符合門檻的標的")
            min_yoy = st.number_input("最低 YoY 門檻 (%)", value=15.0, step=5.0,
                help="月營收年成長率必須超過此門檻") if use_revenue_filter else 15.0
        with rev_col2:
            require_consecutive = st.checkbox("要求連續兩個月達標", value=True,
                help="過去連續兩個月 YoY 均需超過門檻，避免單月爆量") if use_revenue_filter else False
            use_gm_filter = st.checkbox("啟用毛利率擴張篩選（第二道鎖）", value=False,
                help="同時要求最新季毛利率 > 前一季（需抓季財報，速度較慢）")
            use_cashflow_filter = st.checkbox("要求營業現金流為正（地雷股防禦）", value=False,
                help="過濾掉帳面賺錢但營業現金流為負的公司（可能塞貨或做假帳）。抓不到資料的不誤殺。")
        with rev_col3:
            if use_revenue_filter:
                st.info("雷達會先用月營收 YoY 對板塊排分，再挑出個股中通過門檻的標的送進 SMC 狙擊系統。"
                        "\n\n掃描只讀本地快取，不打 FinMind，所以板塊數量不受 API 限制。"
                        "基本面資料請定期用 update_all.py 更新。")

    if st.button("執行全自動動能掃描與派單", type="primary", use_container_width=True):
        st.session_state['radar_dfs'] = {}
        
        # 1. 取得大池子
        candidate_tickers = []
        for sec in r_sectors:
            candidate_tickers.extend(SECTOR_MAP.get(sec, []))
        candidate_tickers = list(set(candidate_tickers)) 
        
        target_tickers = []
        
        # ==========================================
        # 🚦 邏輯分流：AI 雷達過濾 vs 手動全掃描
        # ==========================================
        if use_auto_radar:

            # ── Step 1：月營收對板塊評分，選出最強 N 個板塊 ──
            # 這步只呼叫 FinMind API，不讀任何 K 線
            with st.spinner(f"Step 1／3：正在用月營收 YoY 對 {len(r_sectors)} 個板塊評分..."):
                # 先嘗試月營收評分（不需要 K 線）
                from sector_radar import SectorRadar
                radar = SectorRadar(SECTOR_MAP)
                top_sector_tickers = radar.get_top_sectors_tickers(
                    raw_data_dict={},
                    top_n=top_n_sectors,
                    use_revenue=True,
                )

            # 降級：月營收不足時才讀 K 線做股價動能評分
            if not top_sector_tickers:
                st.warning("月營收資料不足，降級為股價動能模式，正在讀取 K 線...")
                raw_data_dict = {}
                prog = st.progress(0)
                for idx, t in enumerate(candidate_tickers):
                    prog.progress((idx + 1) / len(candidate_tickers))
                    df_radar = fetch_stock_data(t, days=40)
                    if df_radar is not None and not df_radar.empty:
                        raw_data_dict[t] = df_radar
                prog.empty()
                top_sector_tickers = radar.get_top_sectors_tickers(
                    raw_data_dict=raw_data_dict,
                    top_n=top_n_sectors,
                    use_revenue=False,
                )

            if not top_sector_tickers:
                top_sector_tickers = candidate_tickers
                st.warning("雷達無法評分，使用全部候選標的。")

            st.info(f"Step 1 完成：最強 {top_n_sectors} 個板塊共 {len(top_sector_tickers)} 檔標的進入第二關。")

            # ── Step 2：月營收 YoY 個股篩選（第一道鎖）──
            if use_revenue_filter:
                with st.spinner(f"Step 2／3：正在篩選 {len(top_sector_tickers)} 檔標的的月營收 YoY..."):
                    rev_df = screen_sector_by_revenue(
                        top_sector_tickers,
                        min_yoy=min_yoy,
                        require_consecutive=require_consecutive,
                        require_gm_expanding=use_gm_filter,
                        require_positive_cashflow=use_cashflow_filter,
                        cache_only=True,
                    )

                passed = rev_df[rev_df["pass_filter"] == True]["ticker"].tolist()

                # 顯示個股篩選明細
                st.markdown("#### 月營收動能篩選結果")
                st.caption("MoM = 月增率（本月 vs 上月）。僅供參考，不是進場條件——台股月營收有季節性，"
                           "看 YoY 為主，MoM 連續下滑可當轉折警訊。")
                display_rev = rev_df[["ticker", "latest_yoy", "avg_yoy_3m", "mom", "yoy_streak",
                                       "consecutive_growth", "latest_gm", "gm_expanding", "ocf_positive", "pass_filter"]].copy()
                display_rev.columns = ["標的", "最新YoY(%)", "近3月均YoY(%)", "MoM月增(%)", "連續成長月數",
                                        "連兩月達標", "最新毛利率(%)", "毛利擴張", "現金流為正", "通過篩選"]
                st.dataframe(
                    display_rev.style.apply(
                        lambda row: ["background-color: #004d40; color: white;" if row["通過篩選"] else "" for _ in row],
                        axis=1
                    ),
                    use_container_width=True,
                )
                st.caption(f"通過篩選：{len(passed)} / {len(top_sector_tickers)} 檔")

                if passed:
                    target_tickers = passed
                    st.success(f"Step 2 完成：{len(target_tickers)} 檔通過月營收篩選，進入 SMC 狙擊。")

                    # ── 三大法人買賣超（籌碼面參考，只對通過篩選的少數股抓）──
                    with st.expander("三大法人買賣超（籌碼面參考，非進場條件）", expanded=False):
                        st.caption("外資連續買超 + FVG 訊號可作雙重確認。籌碼面僅供參考，不影響進場篩選。")
                        try:
                            from revenue_fetcher import fetch_institutional_net
                            chip_rows = []
                            for tk in target_tickers:
                                inst = fetch_institutional_net(tk.replace(".TW", ""))
                                if inst.get("data_ok"):
                                    chip_rows.append({
                                        "標的": tk,
                                        "外資買賣超(張)": inst.get("foreign_net"),
                                        "外資連買天數": inst.get("foreign_streak"),
                                        "近5日三大法人(張)": inst.get("total_net_5d"),
                                    })
                            if chip_rows:
                                chip_df = pd.DataFrame(chip_rows).sort_values("外資買賣超(張)", ascending=False)
                                st.dataframe(chip_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("無法取得籌碼面資料（可能該批標的無三大法人資料）。")
                        except Exception as e:
                            st.warning(f"籌碼面資料抓取失敗：{e}")
                else:
                    st.warning("無標的通過月營收篩選，放寬條件或關閉此篩選。保留板塊雷達結果繼續掃描。")
                    target_tickers = top_sector_tickers
            else:
                target_tickers = top_sector_tickers
                st.success(f"Step 2 跳過（月營收篩選已關閉）：{len(target_tickers)} 檔進入 SMC 狙擊。")

            if not target_tickers:
                target_tickers = candidate_tickers
        else:
            target_tickers = candidate_tickers
            st.success(f"手動模式：直接鎖定指定的 {len(target_tickers)} 檔標的進行 SMC 狙擊。")

        # ==========================================
        
        with st.spinner("正在檢查大盤宏觀環境..."):
            m_data = _get_benchmark_data(days=250)
            if m_data is not None and not m_data.empty:
                m_close   = m_data['Close'].iloc[-1]
                m_sma60   = m_data['Close'].rolling(60).mean().iloc[-1]
                m_sma200  = m_data['Close'].rolling(200).mean().iloc[-1]
                m_change  = (m_close / m_data['Close'].iloc[-2] - 1) * 100
                m_vol20   = m_data['Close'].pct_change().rolling(20).std().iloc[-1] * (252**0.5) * 100
                m_trend   = "Bull" if m_close > m_sma200 else "Bear"
                m_volreg  = "HighVol" if m_vol20 > vol_threshold else "LowVol"
                scan_regime = f"{m_trend}_{m_volreg}"

                if m_close < m_sma60:
                    st.error(f"大盤斷路器啟動：{regime_benchmark} 位於季線下方，目前環境極度危險，系統拒絕發出任何買入建議。")
                    st.stop()
                if m_change < -2.0:
                    st.warning(f"市場恐慌警告：今日 {regime_benchmark} 跌幅達 {m_change:.2f}%，建議放棄所有新掛單。")

                # Regime 過濾：當前環境不在允許清單就直接靜音
                if scan_allowed_regimes and scan_regime not in scan_allowed_regimes:
                    st.warning(f"當前市場環境 **{scan_regime}**（基準：{regime_benchmark}）不在允許進場的 Regime 清單內，系統今日靜音。"
                               f"\n\n如需強制掃描，請在上方「允許發出訊號的 Regime」加入 {scan_regime}，或更換判斷基準。")
                    st.stop()
                else:
                    st.info(f"當前市場環境：**{scan_regime}**（基準：{regime_benchmark}），符合進場條件，開始掃描。")
            else:
                scan_regime = "Unknown"

        radar_results = []
        progress_bar = st.progress(0)
        seen_tickers = []   # 供相關性過濾使用

        with st.spinner(f"Step 3／3：SMC 引擎正在讀取 {len(target_tickers)} 檔 K 線並狙擊訊號..."):
            for i, ticker in enumerate(target_tickers):
                progress_bar.progress((i + 1) / len(target_tickers))
                try:
                    df = fetch_stock_data(ticker, days=150)
                    if df is None or df.empty or len(df) < 65:
                        continue

                    df_logic = apply_smc_logic(
                        df.copy(),
                        min_gap_pct=r_gap,
                        use_sma_filter=r_sma,
                        use_vol_filter=r_vol,
                        use_atr_filter=r_atr,
                        use_bos_filter=r_bos
                    )

                    last_row = df_logic.iloc[-1]

                    if last_row['FVG']:
                        vol_ma20  = df['Volume'].rolling(20).mean().iloc[-1]
                        vol_surge = safe_divide(df['Volume'].iloc[-1], vol_ma20)
                        entry_price = last_row['Low']
                        sl_price    = entry_price - last_row['adjusted_risk']

                        # 停利目標改用 UI 設定的移動停利 %，不再寫死 10%
                        tp_price = entry_price * (1 + trailing_stop_pct)
                        rr_ratio = safe_divide((tp_price - entry_price), last_row['adjusted_risk'])

                        if vol_surge >= min_momentum and rr_ratio >= min_rr:

                            # 相關性過濾：與已發出訊號的標的板塊相同時限制數量
                            ticker_sector = INDUSTRY_DICT_REVERSE.get(ticker, "其他")
                            same_sector_count = sum(
                                1 for t in seen_tickers
                                if INDUSTRY_DICT_REVERSE.get(t, "") == ticker_sector
                            )
                            if same_sector_count >= 2:
                                continue  # 同板塊已有 2 個訊號，跳過

                            risk_amount      = live_equity * live_risk_pct
                            risk_per_share   = max(entry_price - sl_price, 0.01)
                            recommended_shares = int(risk_amount / risk_per_share)

                            capital_required   = recommended_shares * entry_price
                            max_cash_allowed   = live_equity * max_capital_per_trade

                            if capital_required > max_cash_allowed:
                                recommended_shares = int(max_cash_allowed / entry_price)
                                capital_required   = recommended_shares * entry_price

                            if recommended_shares <= 0:
                                continue

                            seen_tickers.append(ticker)

                            _chip_net, _chip_streak = "—", "—"
                            try:
                                from revenue_fetcher import fetch_institutional_net
                                _c = fetch_institutional_net(ticker.replace(".TW", ""))
                                if _c.get("data_ok"):
                                    _chip_net = _c.get("foreign_net")
                                    _chip_streak = _c.get("foreign_streak")
                            except Exception:
                                pass
                            radar_results.append({
                                "股票代碼": ticker,
                                "股票名稱": STOCK_NAMES.get(ticker, STOCK_NAMES.get(ticker.replace(".TW",""), "—")),
                                "板塊": ticker_sector,
                                "Regime": scan_regime,
                                "動能分數": round(vol_surge, 2),
                                "外資買賣超(張)": _chip_net,
                                "外資連買天數": _chip_streak,
                                "預期賠率(R)": round(rr_ratio, 1),
                                "進場價": round(entry_price, 2),
                                "防彈止損價": round(sl_price, 2),
                                "獲利目標": round(tp_price, 2),
                                "建議買入股數": recommended_shares,
                                "預估耗用資金": round(capital_required, 2)
                            })

                            st.session_state['radar_dfs'][ticker] = df_logic

                            tg_msg = (
                                f"*[SMC 機構派單中心] 鎖定獵物！*\n"
                                f"*標的*：{ticker} ({ticker_sector}) | *環境*：{scan_regime}\n"
                                f"*動能*：{vol_surge:.2f}x | *賠率*：{rr_ratio:.1f} R\n"
                                f"➖➖➖➖➖➖➖➖\n"
                                f"*進場掛單*：`{entry_price:.2f}`\n"
                                f"*防彈止損*：`{sl_price:.2f}`\n"
                                f"*獲利目標*：`{tp_price:.2f}`（移動停利 {trailing_stop_pct*100:.0f}%）\n"
                                f"*建議買入股數*：`{recommended_shares}` 股\n"
                                f"*預估耗用資金*：${capital_required:,.0f}"
                            )
                            send_telegram_message(tg_msg)

                except Exception as e:
                    continue

        progress_bar.empty()
        
        if radar_results:
            radar_df = pd.DataFrame(radar_results).sort_values("動能分數", ascending=False)
            st.session_state['radar_results'] = radar_df 
            st.success(f"SMC 狙擊完成！發現 {len(radar_df)} 檔完美目標。")
        else:
            st.session_state['radar_results'] = pd.DataFrame()
            st.warning(f"今日收盤無符合嚴格條件的標的。今日空手，享受生活保護資金！")

    if 'radar_results' in st.session_state and not st.session_state['radar_results'].empty:
        radar_df = st.session_state['radar_results']
        
        st.subheader("今日實盤派單清單（系統訊號股，含外資籌碼面參考）")
        st.caption("外資買賣超/連買天數為籌碼面參考，協助你在訊號股中排下單優先序，不影響系統篩選。")
        styled_df = radar_df.style.format({
            '動能分數': '{:.2f}', '預期賠率(R)': '{:.2f} R',
            '進場價': '${:.2f}', '防彈止損價': '${:.2f}', '獲利目標': '${:.2f}',
            '建議買入股數': '{:,.0f} 股', '預估耗用資金': '${:,.0f}'
        }, na_rep="—").map(lambda x: "background-color: #004d40; color: white;", subset=['動能分數', '建議買入股數'])
        
        st.dataframe(styled_df, hide_index=True, use_container_width=True)
        st.info("操作建議：請直接打開券商 App，依照「建議買入股數」與「進場掛單價」進行限價掛單。")
        
        st.divider()
        st.markdown("### 交易員肉眼複核 (SMC 結構圖)")
        
        selected_ticker = st.selectbox("選擇要展開 K 線圖的標的：", radar_df['股票代碼'].tolist())
        
        if selected_ticker and selected_ticker in st.session_state['radar_dfs']:
            df_plot = st.session_state['radar_dfs'][selected_ticker]
            trade_info = radar_df[radar_df['股票代碼'] == selected_ticker].iloc[0]
            
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
                low=df_plot['Low'], close=df_plot['Close'], name='K線'
            ))
            
            fvg_signals = df_plot[df_plot['FVG'] == True]
            for idx, row in fvg_signals.iterrows():
                idx_num = df_plot.index.get_loc(idx)
                if idx_num >= 2:
                    fvg_top = row['Low']
                    fvg_bottom = df_plot['High'].iloc[idx_num - 2]
                    end_idx = min(idx_num + 10, len(df_plot) - 1) 
                    
                    fig.add_shape(
                        type="rect",
                        x0=df_plot.index[idx_num-2], y0=fvg_bottom,
                        x1=df_plot.index[end_idx], y1=fvg_top,
                        fillcolor="rgba(0, 255, 128, 0.2)",
                        line=dict(width=0), layer="below"
                    )

            if 'SMA60' in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot['SMA60'], 
                    mode='lines', name='SMA 60', line=dict(color='orange', width=1)
                ))

            if 'Swing_High_20' in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot['Swing_High_20'], 
                    mode='lines', name='20日轉折高點', line=dict(color='gray', width=1, dash='dot')
                ))

            fig.add_hline(y=trade_info['進場價'], line_dash="solid", line_color="cyan", annotation_text="預計進場價 (FVG邊緣)")
            fig.add_hline(y=trade_info['防彈止損價'], line_dash="dash", line_color="red", annotation_text="防彈止損 (SL)")
            fig.add_hline(y=trade_info['獲利目標'], line_dash="dash", line_color="green", annotation_text="獲利目標 (TP)")

            fig.update_layout(
                template="plotly_dark",
                title=f"{selected_ticker} - SMC 結構觀測窗",
                yaxis_title="價格",
                xaxis_rangeslider_visible=False, 
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
#  分頁 2：單股深度回測實驗室
# ==========================================
with tab2:
    st.header("回測(單股)")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    bt_t = c1.selectbox("標的選擇", SEARCH_LIST, key="b_t").split(" - ")[0]
    bt_years = c2.slider("回測年限", 1, 15, 10, key="b_y")
    bt_mode = c3.selectbox("出場方式", ["固定盈虧比", "移動停利"], key="b_m")
    bt_val = c4.number_input("RR / 停利%", value=10.0, key="b_v")
    
    st.markdown("##### 真實資金與 SMC 濾網設定")
    col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
    initial_cap = col_c1.number_input("初始本金 (NTD)", value=1000000, step=100000, key="b_cap")
    risk_pct = col_c2.number_input("單筆風險 (%)", value=2.0, step=0.5, key="b_risk") / 100.0
    bt_vol = col_c3.checkbox("要求起漲爆量", value=False, key="bt_vol")
    bt_atr = col_c4.checkbox("強勢實體位移", value=False, key="bt_atr")
    bt_bos = col_c5.checkbox("突破近期新高", value=False, key="bt_bos")

    st.markdown("##### 實戰摩擦力設定")
    col_b1, col_b2, col_b3, col_b4, col_b5, col_b6 = st.columns(6)
    bt_gap = col_b1.number_input("最小缺口(%)", value=0.3, step=0.1, key="b_g")
    bt_sma = col_b2.checkbox("啟用季線濾網", value=True, key="b_s")
    fee_disc = col_b3.number_input("手續費折讓", value=0.28, key="b_f")
    slippage_pct = col_b4.number_input("滑價(%)", value=0.1, step=0.05, key="b_slip")
    min_vol = col_b5.number_input("最低成交量", value=1000, step=500, key="b_vol2")
    oos_ratio = col_b6.slider("樣本外比例(%)", 10, 50, 30, step=10, key="b_oos")

    col_btn1, col_btn2 = st.columns(2)
    run_normal = col_btn1.button("執行科學回測 (含 Monte Carlo)", type="primary", use_container_width=True)
    run_heatmap = col_btn2.button("執行參數壓力測試 (Heatmap)", use_container_width=True)

    df_raw = fetch_stock_data(bt_t, days=bt_years * 365)

    if run_normal and not df_raw.empty:
        df = apply_smc_logic(df_raw.copy(), bt_gap, bt_sma, use_vol_filter=bt_vol, use_atr_filter=bt_atr, use_bos_filter=bt_bos)
        split_idx = int(len(df) * (1 - oos_ratio/100))
        df_is, df_oos = df.iloc[:split_idx], df.iloc[split_idx:]

        # 載入該標的的歷史基本面序列（供出場判定用）
        _bt_fundamental = {}
        try:
            from revenue_fetcher import fetch_revenue_history, fetch_margin_history
            _ticker_pure = str(bt_t).replace(".TW", "")
            _bt_fundamental = {
                bt_t: {
                    "rev":    fetch_revenue_history(_ticker_pure),
                    "margin": fetch_margin_history(_ticker_pure),
                }
            }
        except Exception:
            pass

        res_is  = run_backtest_core(df_is,  bt_mode, bt_val, slippage_pct, fee_disc, 0.003, min_vol,
                                    ticker=bt_t, fundamental_cache=_bt_fundamental)
        res_oos = run_backtest_core(df_oos, bt_mode, bt_val, slippage_pct, fee_disc, 0.003, min_vol,
                                    ticker=bt_t, fundamental_cache=_bt_fundamental)
        
        df_is_res = res_is[0] if (res_is and len(res_is) > 0) else pd.DataFrame()
        df_oos_res = res_oos[0] if (res_oos and len(res_oos) > 0) else pd.DataFrame()

        # 在 UI 端建立統計數據解碼器
        def calc_metrics(res_df):
            if res_df is None or res_df.empty or '真實R' not in res_df.columns:
                return 0, 0.0, 0.0, 0.0, 0.0
            t = len(res_df)
            w = len(res_df[res_df['真實R'] > 0]) / t if t > 0 else 0
            avg_w = res_df[res_df['真實R'] > 0]['真實R'].mean() if w > 0 else 0
            avg_l = abs(res_df[res_df['真實R'] < 0]['真實R'].mean()) if w < 1 else 1
            exp = (w * avg_w) - ((1 - w) * avg_l)
            eq = res_df['真實R'].cumsum()
            mdd = abs((eq - eq.expanding().max()).min()) if not eq.empty else 0.0
            r_std = res_df['真實R'].std()
            # 簡單計算 R 倍數的夏普值
            sharpe = (res_df['真實R'].mean() / r_std) * np.sqrt(min(t, 252)) if t > 1 and r_std != 0 else 0.0
            return t, w, exp, mdd, sharpe

        # 執行在地運算，取得 UI 顯示用的 5 大指標
        t_is, w_is, exp_is, mdd_is, sharpe_is = calc_metrics(df_is_res)
        t_oos, w_oos, exp_oos, mdd_oos, sharpe_oos = calc_metrics(df_oos_res)
        
        st.markdown(f"### OOS 盲測對比")
        c_is, c_oos = st.columns(2)
        with c_is:
            st.success(f"**訓練期 (IS)**: {exp_is:.2f} R")
            st.write(f"夏普值: {sharpe_is:.2f}")
        with c_oos:
            st.warning(f"**盲測期 (OOS)**: {exp_oos:.2f} R")
            st.write(f"夏普值: {sharpe_oos:.2f}")

        # ==========================================
        # 🌟 交易明細渲染與 Excel CSV 無損下載引擎
        # ==========================================
        st.markdown("### 盲測期 (OOS) 真實交易明細")
        if not df_oos_res.empty:
            # 渲染明細表
            st.dataframe(df_oos_res, use_container_width=True, hide_index=True)
            
            # 建立 CSV 快取並轉換編碼，避免 Excel 開啟中文亂碼
            csv_data = df_oos_res.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="下載完整交易明細至 Excel (CSV)",
                data=csv_data,
                file_name=f"SMC_{bt_t}_OOS_Trades.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        else:
            st.info("盲測期間沒有觸發任何交易。")

        st.divider()
        st.subheader("蒙地卡羅壓力測試 (1,000 次序列重抽)")
        mc_results = run_monte_carlo(df_oos_res, risk_per_trade_pct=risk_pct * 100)
        
        if mc_results:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("中位數預期獲利", f"{mc_results['median_final_r']:.2f} R")
            m2.metric("破產機率 (DD > 30%)", f"{mc_results['prob_of_ruin_pct']:.1f}%", delta_color="inverse")
            m3.metric("中位數最大回撤", f"{mc_results['median_max_dd_r']:.2f} R")
            m4.metric("極端最慘回撤", f"{mc_results['worst_max_dd_r']:.2f} R")
        else:
            st.info("交易次數不足，無法執行蒙地卡羅模擬。")

# ==========================================
#  分頁 3：投資組合 Alpha 驗證引擎
# ==========================================
with tab3:
    st.header("投資組合 Alpha 驗證")
    
    p_col1, p_col2 = st.columns([1, 4])
    with p_col1:
        p_sector = st.multiselect("測試板塊", list(INDUSTRY_DICT.keys()), default=[list(INDUSTRY_DICT.keys())[0]])
        p_years = st.slider("回測年限", 1, 15, 10, key="py_slider")
        p_cap = st.number_input("初始本金", value=1000000, step=100000, key="pcap_input")
        p_risk = st.slider("單筆風險 (%)", 0.5, 5.0, 2.0, 0.5, key="prisk_slider")
        p_max_pos = st.slider("最大持倉數", 1, 20, 5)
        p_max_sec = st.slider("單一產業上限", 1, 5, 2)
        p_max_position = st.slider(
            "單股部位上限 (%)", 10, 100, 15, 5,
            help="單一股票最多佔帳戶幾%。預設15%。放寬會讓部位更集中、回撤放大。"
                 "WFO會自動沿用這個值,確保盲測與單次回測一致。",
        ) / 100.0
        p_account_dd = st.slider("帳戶停機線 (%)", 10, 50, 20, 5) / 100.0

        st.divider()
        st.markdown("###### 市場環境過濾")
        ALL_REGIMES = ["Bull_LowVol", "Bull_HighVol", "Bear_LowVol", "Bear_HighVol"]
        p_regimes = st.multiselect(
            "允許進場的 Regime（空白 = 全部）",
            ALL_REGIMES,
            default=["Bull_LowVol", "Bear_HighVol"],
            help="根據 Regime Analysis 結果，只在正期望值環境進場",
        )
        # Regime 基準與門檻（與 Tab1 掃描保持一致才能對照）
        _p_bench_label = st.selectbox(
            "Regime 判斷基準",
            ["0050 權值股", "加權指數 (全市場)", "櫃買指數 (中小型股)"],
            index=0,
            key="p_bench_sel",
            help="回測用的基準要和 Tab1 掃描一致，回測結論才能套用到實際掃描",
        )
        _p_bench_map = {"0050 權值股": "0050.TW", "加權指數 (全市場)": "TAIEX", "櫃買指數 (中小型股)": "TPEx"}
        p_regime_benchmark = _p_bench_map[_p_bench_label]
        p_vol_threshold = st.slider(
            "高波動門檻 (年化波動率 %)",
            min_value=15, max_value=50, value=25, step=5,
            key="p_vol_thresh",
            help="與 Tab1 設定一致。改這個會改變 HighVol/LowVol 的判定",
        )
        p_min_holding = st.slider(
            "最短預期持倉天數",
            min_value=0, max_value=30, value=10, step=5,
            help="預期持倉天數低於此值的訊號直接跳過，過濾短線雜訊（0 = 不過濾）",
        )

        st.markdown("###### 進場基本面過濾（與實盤掃描一致）")
        _use_entry_yoy = st.checkbox(
            "進場要求月營收 YoY 達標", value=False, key="p_entry_yoy_chk",
            help="回測進場時要求當下月營收 YoY ≥ 門檻（無前視偏差，用歷史序列點時間查詢）。"
                 "勾選後回測會和實盤的月營收篩選一致。",
        )
        p_entry_min_yoy = None
        if _use_entry_yoy:
            p_entry_min_yoy = st.slider("進場 YoY 門檻 (%)", 0, 50, 15, 5, key="p_entry_yoy_val")
        p_entry_require_gm = st.checkbox(
            "進場要求毛利率擴張", value=False, key="p_entry_gm_chk",
            help="回測進場時要求最新季毛利率 > 前一季（無前視偏差）。",
        )
        p_entry_require_ocf = st.checkbox(
            "進場要求營業現金流為正（地雷股防禦）", value=False, key="p_entry_ocf_chk",
            help="回測進場時要求最新季營業現金流 > 0（無前視偏差，季報公告延遲）。"
                 "抓不到資料的標的不誤殺，與實盤掃描端邏輯一致。",
        )

        st.divider()
        p_mode = st.selectbox("出場方式", ["固定盈虧比", "移動停利"], key="pmode_sel")
        p_val = st.number_input("目標倍數/停利%", value=10.0, key="pval_num")
        p_gap = st.number_input("最小缺口(%)", value=0.0, step=0.1, key="pgap_num")
        p_sma = st.checkbox("季線濾網", value=True, key="psma_chk")

        st.markdown("###### SMC 嚴格選配模組")
        p_vol = st.checkbox("要求起漲爆量", value=False, key="pvol_chk")
        p_atr = st.checkbox("強勢實體位移", value=False, key="patr_chk")
        p_bos = st.checkbox("突破近期新高", value=False, key="pbos_chk")

        st.markdown("###### 動態板塊輪動（最貼近實盤）")
        p_rotation = st.checkbox(
            "啟用動態板塊輪動", value=False, key="p_rotation_chk",
            help="每月用當下已公告的月營收，自動選最強 N 個板塊，只在這些板塊進場。"
                 "這與你實盤掃描的板塊雷達邏輯一致（point-in-time，無前視偏差）。"
                 "啟用後會自動用全市場板塊評分，不受上方手選板塊限制。",
        )
        p_rotation_n = 3
        if p_rotation:
            p_rotation_n = st.slider("輪動選最強幾個板塊", 1, 5, 3, key="p_rotation_n")

        run_portfolio = st.button("執行組合驗證", type="primary", use_container_width=True)

    with p_col2:
        if run_portfolio:
            if not p_sector and not p_rotation:
                st.warning("請至少選擇一個板塊（或啟用動態板塊輪動）！")
            else:
                # 全板塊地圖（輪動評分用，涵蓋全市場）
                full_sector_map = {sec: tks for sec, tks in INDUSTRY_DICT.items()}

                if p_rotation:
                    # 動態輪動：標的池用全市場（輪動會每月自動篩選最強板塊）
                    tickers_to_test = list({t for tks in INDUSTRY_DICT.values() for t in tks})
                else:
                    # 固定板塊：只測手選的板塊
                    tickers_to_test = []
                    for sec in p_sector:
                        tickers_to_test.extend(INDUSTRY_DICT.get(sec, []))
                    tickers_to_test = list(set(tickers_to_test))
                sector_map = {ticker: sec for sec, tickers in INDUSTRY_DICT.items() for ticker in tickers}

                with st.spinner(f"正在平行運算 {len(tickers_to_test)} 檔標的，請稍候..."):
                    cfg = PortfolioConfig(
                        tickers=tickers_to_test, years=p_years, bt_mode=p_mode, bt_val=p_val,
                        gap_pct=p_gap, use_sma=p_sma, use_vol_filter=p_vol, use_atr_filter=p_atr,
                        use_bos_filter=p_bos, slippage_pct=0.002, fee_disc=0.28, min_vol=1000,
                        max_concurrent=p_max_pos, max_per_sector=p_max_sec, sector_map=sector_map,
                        risk_pct=p_risk, initial_capital=p_cap, account_dd_limit=p_account_dd,
                        max_position_pct=p_max_position,
                        allowed_regimes=p_regimes if p_regimes else None,
                        min_holding_days=p_min_holding,
                        regime_benchmark=p_regime_benchmark,
                        vol_threshold=float(p_vol_threshold),
                        entry_min_yoy=p_entry_min_yoy,
                        entry_require_gm=p_entry_require_gm,
                        entry_require_ocf=p_entry_require_ocf,
                        enable_sector_rotation=p_rotation,
                        sector_top_n=p_rotation_n,
                        full_sector_map=full_sector_map,
                    )
                    p_df, p_stats, bm_df = get_cached_portfolio_results(cfg)
                    
                    if p_stats and not p_df.empty:
                        st.session_state['p_df'] = p_df
                        st.session_state['p_stats'] = p_stats
                        st.session_state['bm_df'] = bm_df
                        st.session_state['p_cap'] = p_cap
                        # 供 Tab 5 WFO 使用
                        st.session_state['p_tickers']    = tickers_to_test
                        st.session_state['p_bt_mode']    = p_mode
                        st.session_state['p_risk']       = p_risk
                        st.session_state['p_account_dd'] = p_account_dd
                        st.session_state['p_max_pos']    = p_max_pos
                        # 單股上限：若這版有該滑桿(p_max_position)才存，沒有就跳過(WFO會用預設0.15)
                        if 'p_max_position' in locals():
                            st.session_state['p_max_position'] = p_max_position
                        st.session_state['p_max_sec']    = p_max_sec
                        # ✅ 進場過濾也存起來，讓 WFO 用「跟單次回測一樣的策略」驗證
                        st.session_state['p_entry_min_yoy']     = p_entry_min_yoy
                        st.session_state['p_entry_require_gm']  = p_entry_require_gm
                        st.session_state['p_entry_require_ocf'] = p_entry_require_ocf
                        st.session_state['p_regimes']           = p_regimes if p_regimes else None
                        st.session_state['p_rotation']          = p_rotation
                        st.session_state['p_rotation_n_val']    = p_rotation_n
                        st.session_state['p_full_sector_map']   = full_sector_map
                    else:
                        st.error("引擎未觸發交易。請嘗試調低「最小缺口」或取消勾選「嚴格選配模組」。")

    # ==========================================
    # 將報表呈現區移出側邊欄，享受全螢幕寬度！
    # ==========================================
    if 'p_df' in st.session_state and not st.session_state['p_df'].empty:
        p_df = st.session_state['p_df']
        p_stats = st.session_state['p_stats']
        bm_df = st.session_state.get('bm_df', pd.DataFrame()) 
        p_cap_state = st.session_state.get('p_cap', 1000000)  
        
        st.divider()
        st.markdown(f"### 投資組合綜合績效報表 (初始本金：NT$ {p_cap_state:,})")
        
        # 第一層：資金與風險宏觀指標 (Macro KPIs)
        st.markdown("##### 帳戶宏觀指標 (Macro KPIs)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("期末總淨值", f"NT$ {p_stats.get('期末總淨值(NTD)', 0):,}")
        m2.metric("真實總報酬", f"{p_stats.get('真實總報酬(%)', 0)} %")
        m3.metric("最大回撤 (MDD)", f"-{p_stats.get('真實最大回撤(%)', 0)} %")
        
        sharpe_val = p_stats.get('夏普值(Sharpe)', 0)
        sharpe_str = str(sharpe_val)
        if isinstance(sharpe_val, (int, float)):
            if sharpe_val >= 1.5: sharpe_str += " (極佳)"
            elif sharpe_val >= 1.0: sharpe_str += " (優秀)"
        m4.metric("夏普比率 (Sharpe)", sharpe_str)
        
        st.write("") 
        
        # 第二層：微觀交易品質指標 (Trade Quality)
        st.markdown("##### 交易品質指標 (Trade Quality)")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("勝率 (Win Rate)", f"{p_stats.get('勝率(%)', 0)} %")
        t2.metric("獲利因子 (Profit Factor)", f"{p_stats.get('獲利因子(PF)', 0)}")
        t3.metric("系統期望值 (Exp)", f"{p_stats.get('Portfolio期望值(R)', 0)} R")
        t4.metric("平均 R 倍數", f"{p_stats.get('平均R倍數', 0)} R")
        t5.metric("總成交 / 淘汰", f"{p_stats.get('總交易次數', 0)} / {p_stats.get('風控與資金不足淘汰次數', 0)}")

        st.subheader(f"資金曲線對決：SMC 策略 vs 0050 大盤")
        fig_p = go.Figure()
        
        fig_p.add_trace(go.Scatter(x=p_df['出場日期'], y=p_df['交易後總淨值'], mode='lines', name='SMC 策略 (真資金)', line=dict(color='cyan', width=2)))
        
        if not bm_df.empty:
            date_col = 'Date' if 'Date' in bm_df.columns else 'index' if 'index' in bm_df.columns else bm_df.columns[0]
            bm_equity = p_cap_state * (1 + bm_df['基準報酬(%)'] / 100) 
            bm_final = round(bm_equity.iloc[-1], 0)
            fig_p.add_trace(go.Scatter(x=bm_df[date_col], y=bm_equity, mode='lines', name=f'0050 持有 (NT$ {bm_final:,.0f})', line=dict(color='yellow', width=2, dash='dot')))
        
        fig_p.update_layout(template="plotly_dark", hovermode="x unified", height=500)
        st.plotly_chart(fig_p, use_container_width=True)
        
        st.divider()
        st.markdown("### 避險基金級深度診斷 (Hedge Fund Analytics)")
        
        fig_dd = px.area(p_df, x='出場日期', y='真實回撤(%)', title='水下回撤曲線 (Underwater Curve)', height=300)
        fig_dd.update_traces(fillcolor='rgba(255, 0, 0, 0.3)', line=dict(color='red', width=1))
        fig_dd.update_layout(template="plotly_dark", yaxis_title='距離歷史高點回撤 (%)', hovermode="x unified")
        st.plotly_chart(fig_dd, use_container_width=True)
        
        st.markdown("####贏家集中度與年度拆解")
        top_5_trades = p_df.nlargest(5, '絕對損益')
        total_net_profit = p_df['絕對損益'].sum()
        top_5_profit = top_5_trades['絕對損益'].sum()
        dependency_pct = (top_5_profit / total_net_profit) * 100 if total_net_profit > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("前 5 筆神單總利潤", f"NT$ {top_5_profit:,.0f}")
        c2.metric("極端值利潤占比", f"{round(dependency_pct, 1)} %", help="如果超過 30%，代表策略極度不穩定，依賴運氣。")
        
        p_df['年份'] = pd.to_datetime(p_df['出場日期']).dt.year
        yearly_stats = p_df.groupby('年份')['絕對損益'].sum().reset_index()
        fig_year = px.bar(yearly_stats, x='年份', y='絕對損益', text='絕對損益', title='各年度絕對淨利 (NTD)', color='絕對損益', color_continuous_scale=['#ff3333', '#222222', '#00ffcc'])
        fig_year.update_traces(texttemplate='NT$ %{text:,.0f}', textposition='outside')
        fig_year.update_layout(template="plotly_dark", yaxis_title='利潤 (NTD)')
        st.plotly_chart(fig_year, use_container_width=True)

        with st.expander("查看這 5 筆極端值交易明細"):
            st.dataframe(top_5_trades[['日期', '出場日期', '標的', '板塊', 'Regime', '真實R', '絕對損益']], hide_index=True)

        # ==========================================
        # 新增：全市場交易明細表與 CSV 匯出功能
        # ==========================================
        st.divider()
        st.markdown("### 全市場歷史交易明細 (Full Trade Log)")
        
        # 使用 Streamlit 渲染 DataFrame (預設支援排序與搜尋)
        st.dataframe(p_df, use_container_width=True, hide_index=True)
        
        # 轉換為 utf-8-sig 確保 Excel 開啟中文不會亂碼
        csv_data = p_df.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            label="下載完整投資組合交易明細至 Excel (CSV)",
            data=csv_data,
            file_name="SMC_Portfolio_Full_Trades.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

        # ==========================================
        # 蒙地卡羅模擬
        # ==========================================
        st.divider()
        st.markdown("#### 蒙地卡羅破產模擬 (真・複利版)")
        st.caption("將歷史交易順序打亂重抽，並以真實複利(Compounding)計算淨值，測試連虧地獄。")
        
        trade_results = p_df['真實R'].tolist()
        
        if len(trade_results) < 20:
            st.warning("實際交易樣本數低於 20 筆，執行蒙地卡羅模擬易產生「統計幻覺」，已自動攔截。建議放寬策略參數。")
        else:
            num_simulations = 1000
            sim_lengths = len(trade_results)
            mc_final_returns = []
            mc_max_dds = []
            bankrupt_count = 0
            
            block_size = 5
            blocks = [trade_results[i:i+block_size] for i in range(len(trade_results)-block_size+1)]
            if not blocks: blocks = [[r] for r in trade_results] 
            
            np.random.seed(42)
            base_risk_frac = st.session_state.get('prisk_slider', 2.0) / 100.0
            
            for _ in range(num_simulations):
                sim_seq = []
                while len(sim_seq) < sim_lengths:
                    chosen_block = blocks[np.random.randint(0, len(blocks))]
                    sim_seq.extend(chosen_block)
                sim_seq = sim_seq[:sim_lengths]
                
                sim_equity = np.zeros(sim_lengths + 1)
                sim_equity[0] = p_cap_state
                peak_eq = p_cap_state
                
                for i, r in enumerate(sim_seq):
                    current_eq = sim_equity[i]
                    peak_eq = max(peak_eq, current_eq)
                    current_dd = (peak_eq - current_eq) / peak_eq
                    
                    active_risk = base_risk_frac * 0.5 if current_dd > 0.20 else base_risk_frac
                    compounding_base = min(current_eq, 10000000) 
                    
                    trade_profit = compounding_base * active_risk * r
                    sim_equity[i+1] = current_eq + trade_profit
                    
                if np.any(sim_equity <= 0):
                    mc_final_returns.append(-100.0)
                    mc_max_dds.append(100.0)
                    bankrupt_count += 1
                else:
                    mc_final_returns.append((sim_equity[-1]/p_cap_state - 1) * 100)
                    peak_array = np.maximum.accumulate(sim_equity)
                    dd_array = ((peak_array - sim_equity) / peak_array) * 100.0
                    mc_max_dds.append(np.max(dd_array))
                    
            mc_c1, mc_c2, mc_c3 = st.columns(3)
            mc_c1.metric("MC 平均期末報酬", f"{round(np.mean(mc_final_returns), 2)} %")
            mc_c2.metric("95% 最差回撤 (黑天鵝)", f"-{round(np.percentile(mc_max_dds, 95), 2)} %")
            
            ruin_prob = (bankrupt_count / num_simulations) * 100
            if ruin_prob == 0: mc_c3.metric("破產歸零機率", "0.0 %", "完美防禦")
            elif ruin_prob < 5: mc_c3.metric("破產歸零機率", f"{round(ruin_prob, 2)} %", "風險可控", delta_color="off")
            else: mc_c3.metric("破產歸零機率", f"{round(ruin_prob, 2)} %", "危險！調降單筆風險", delta_color="inverse")

        # ── Regime 市場環境分析 ──
        st.divider()
        st.markdown("#### 市場環境分組分析（Regime Analysis）")
        st.caption("找出策略在哪種市場環境最強、哪種環境應該空手。")

        if 'Regime' in p_df.columns and p_df['Regime'].notna().any():
            regime_rows = []
            for regime, grp in p_df.groupby('Regime'):
                t = len(grp)
                wins = len(grp[grp['真實R'] > 0])
                wr   = wins / t if t > 0 else 0
                avg_w = grp[grp['真實R'] > 0]['真實R'].mean() if wins > 0 else 0
                avg_l = abs(grp[grp['真實R'] < 0]['真實R'].mean()) if wins < t else 1
                exp   = (wr * avg_w) - ((1 - wr) * avg_l)
                net_r = grp['真實R'].sum()
                regime_rows.append({
                    'Regime':       regime,
                    '交易筆數':     t,
                    '勝率(%)':      round(wr * 100, 1),
                    '期望值(R)':    round(exp, 3),
                    '總淨利(R)':    round(net_r, 2),
                    '平均獲利(R)':  round(avg_w, 2),
                    '平均虧損(R)':  round(avg_l, 2),
                })
            regime_df = pd.DataFrame(regime_rows).sort_values('期望值(R)', ascending=False)

            # 顏色標記：期望值正 → 綠，負 → 紅
            def _regime_color(val):
                if isinstance(val, (int, float)):
                    if val > 0.3:  return 'background-color: #1b5e20; color: white'
                    if val > 0:    return 'background-color: #2e7d32; color: white'
                    if val < 0:    return 'background-color: #b71c1c; color: white'
                return ''

            st.dataframe(
                regime_df.style.applymap(_regime_color, subset=['期望值(R)', '總淨利(R)']),
                use_container_width=True,
                hide_index=True,
            )

            # 建議
            good_regimes = regime_df[regime_df['期望值(R)'] > 0.2]['Regime'].tolist()
            bad_regimes  = regime_df[regime_df['期望值(R)'] < 0]['Regime'].tolist()

            if good_regimes:
                st.success(f"**建議只在這些環境進場：** {' / '.join(good_regimes)}")
            if bad_regimes:
                st.error(f"**建議空手的環境（期望值為負）：** {' / '.join(bad_regimes)}")

            # 長條圖
            fig_regime = go.Figure()
            colors = ['#00e676' if v > 0 else '#ff5252' for v in regime_df['期望值(R)']]
            fig_regime.add_trace(go.Bar(
                x=regime_df['Regime'],
                y=regime_df['期望值(R)'],
                marker_color=colors,
                text=regime_df['交易筆數'].astype(str) + ' 筆',
                textposition='outside',
                name='期望值 (R)',
            ))
            fig_regime.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.5)
            fig_regime.update_layout(
                template='plotly_dark',
                title='各市場環境期望值（綠=正，紅=負）',
                xaxis_title='Regime',
                yaxis_title='期望值 (R)',
            )
            st.plotly_chart(fig_regime, use_container_width=True)

            # 計算如果只跑正期望值環境的假設績效
            if good_regimes and bad_regimes:
                filtered = p_df[p_df['Regime'].isin(good_regimes)]
                filtered_net = filtered['真實R'].sum()
                full_net     = p_df['真實R'].sum()
                filtered_wr  = len(filtered[filtered['真實R'] > 0]) / len(filtered) * 100 if len(filtered) > 0 else 0
                st.info(
                    f"**假設只在正期望值環境進場：**\n"
                    f"交易筆數從 {len(p_df)} 筆縮減到 {len(filtered)} 筆（{round(len(filtered)/len(p_df)*100,1)}%）\n"
                    f"總淨利 R 從 {round(full_net,1)} → {round(filtered_net,1)} R\n"
                    f"勝率 → {round(filtered_wr,1)}%\n"
                    f"MDD 理論上會大幅改善（需重新回測驗證）"
                )
        else:
            st.info("Regime 欄位不存在，請重新執行回測。")

        # ── 歷史黑天鵝場景壓力測試 ──
        st.divider()
        st.markdown("#### 歷史黑天鵝場景壓力測試")
        st.caption("在你的真實交易序列中插入歷史重大崩盤事件，測試帳戶能否存活。")
        from monte_carlo import run_scenario_stress_test
        scenario_results = run_scenario_stress_test(
            p_df, initial_capital=p_cap_state,
            risk_per_trade_pct=st.session_state.get('prisk_slider', 2.0)
        )
        if scenario_results:
            sc_cols = st.columns(len(scenario_results))
            for col, (key, sc) in zip(sc_cols, scenario_results.items()):
                survived_label = "存活" if sc["survived"] else "破產"
                col.metric(
                    sc["name"],
                    f"最低淨值 NT${sc['min_equity']:,.0f}",
                    f"最大回撤 {sc['max_dd_pct']}% | {survived_label}",
                    delta_color="normal" if sc["survived"] else "inverse",
                    help=sc["desc"],
                )
with tab7:
    st.header("歷史戰績資料庫")
    history_df = load_history()
    if not history_df.empty: st.dataframe(history_df, use_container_width=True, hide_index=True)
    else: st.info("目前尚無存檔。")

# --- 分頁 5：終極防禦與衰減檢定 (Robustness & Alpha Decay) ---
with tab4:
    st.header("終極防禦與衰減檢定 (Hedge Fund Due Diligence)")
    st.caption("驗證參數是否過度最佳化 (Overfitting) 以及 Alpha 優勢是否隨時間衰減。")
    
    st.divider()
    
    # ==========================================
    # 模組 1：Alpha Decay (前向滾動時間衰減檢定)
    # ==========================================
    st.subheader("1. Alpha 衰減檢定 (Walk-Forward Alpha Decay)")
    st.markdown("真正的優勢(Alpha)不該像冰塊一樣快速融化。我們將檢視策略在過去 10 年間，每年的**期望值(R)**與**勝率**是否保持平穩。")
    
    #  透過 session_state 讀取資料，不再受限於 Tab 切換重整問題！
    if 'p_df' in st.session_state and not st.session_state['p_df'].empty:
        decay_df = st.session_state['p_df'].copy()
        decay_df['年份'] = pd.to_datetime(decay_df['出場日期']).dt.year
        
        yearly_metrics = []
        for year, group in decay_df.groupby('年份'):
            total_trades_yr = len(group)
            win_trades_yr = len(group[group['真實R'] > 0])
            win_rate_yr = win_trades_yr / total_trades_yr if total_trades_yr > 0 else 0
            
            avg_w_yr = group[group['真實R'] > 0]['真實R'].mean() if win_rate_yr > 0 else 0
            avg_l_yr = abs(group[group['真實R'] < 0]['真實R'].mean()) if win_rate_yr < 1 else 1
            expectancy_yr = (win_rate_yr * avg_w_yr) - ((1 - win_rate_yr) * avg_l_yr)
            
            yearly_metrics.append({
                '年份': year,
                '交易次數': total_trades_yr,
                '勝率(%)': round(win_rate_yr * 100, 2),
                '期望值(R)': round(expectancy_yr, 2)
            })
            
        wfa_df = pd.DataFrame(yearly_metrics)
        
        from plotly.subplots import make_subplots
        fig_decay = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_decay.add_trace(go.Bar(x=wfa_df['年份'], y=wfa_df['期望值(R)'], name="年度期望值 (R)", marker_color='cyan'), secondary_y=False)
        fig_decay.add_trace(go.Scatter(x=wfa_df['年份'], y=wfa_df['勝率(%)'], name="年度勝率 (%)", mode='lines+markers', line=dict(color='yellow', width=2)), secondary_y=True)
        
        fig_decay.update_layout(template="plotly_dark", title="Alpha 衰減追蹤 (Expectancy vs Win Rate Over Time)", hovermode="x unified")
        fig_decay.update_yaxes(title_text="期望值 (R)", secondary_y=False)
        fig_decay.update_yaxes(title_text="勝率 (%)", secondary_y=True, range=[0, 100])
        
        st.plotly_chart(fig_decay, use_container_width=True)
        
        first_half_r = wfa_df['期望值(R)'].iloc[:len(wfa_df)//2].mean()
        second_half_r = wfa_df['期望值(R)'].iloc[len(wfa_df)//2:].mean()
        st.info(f"**Alpha 衰減診斷：** 前半段平均期望值 **{round(first_half_r, 2)} R** ➔ 後半段平均期望值 **{round(second_half_r, 2)} R**。如果後半段出現斷崖式下跌，代表策略已被市場淘汰。")
    else:
        st.warning("請先至「投資組合 (Tab 3)」執行一次回測，產生數據後再來檢視 Alpha 衰減！")

    st.divider()

    # ==========================================
    # 模組 1.5：出場原因分析
    # ==========================================
    st.subheader("1.5 出場原因分析（技術 vs 基本面）")
    st.markdown("檢視有多少筆交易是由基本面出場（月營收轉負或毛利衰退）觸發，了解兩套出場邏輯的實際貢獻比例。")

    if 'p_df' in st.session_state and not st.session_state['p_df'].empty:
        exit_df = st.session_state['p_df'].copy()

        if 'reason' in exit_df.columns:
            # 分類
            def _classify(r):
                if isinstance(r, str) and r.startswith("Fundamental_Exit"):
                    detail = r.replace("Fundamental_Exit:", "")
                    return f"基本面出場 ({detail})"
                elif r in ("Hit_Stop", "Gap_Through_Stop"):
                    return "停損出場"
                elif r == "Catastrophic_Stop":
                    return "緊急停損出場 (保護期)"
                elif r in ("Hit_Target", "Gap_Over_Target"):
                    return "停利出場"
                elif r == "移動停利":
                    return "移動停利出場"
                else:
                    return f"其他 ({r})" if r else "未知"

            exit_df["出場類型"] = exit_df["reason"].apply(_classify)
            reason_counts = exit_df.groupby("出場類型").agg(
                筆數=("真實R", "count"),
                平均R=("真實R", "mean"),
                總淨利R=("真實R", "sum"),
            ).round(2).reset_index()

            er1, er2 = st.columns([1, 2])
            with er1:
                st.dataframe(reason_counts, use_container_width=True, hide_index=True)
            with er2:
                fig_exit = go.Figure()
                fig_exit.add_trace(go.Bar(
                    x=reason_counts["出場類型"],
                    y=reason_counts["筆數"],
                    name="交易筆數",
                    marker_color=["#00e676" if "基本面" in t else "#448aff" for t in reason_counts["出場類型"]],
                    yaxis="y1",
                ))
                fig_exit.add_trace(go.Scatter(
                    x=reason_counts["出場類型"],
                    y=reason_counts["平均R"],
                    name="平均 R",
                    mode="markers+lines",
                    line=dict(color="yellow", width=2),
                    yaxis="y2",
                ))
                fig_exit.update_layout(
                    template="plotly_dark",
                    title="各出場類型：筆數 vs 平均 R",
                    yaxis=dict(title="筆數"),
                    yaxis2=dict(title="平均 R", overlaying="y", side="right"),
                    legend=dict(x=0.01, y=0.99),
                )
                st.plotly_chart(fig_exit, use_container_width=True)

            fund_pct = round(
                len(exit_df[exit_df["出場類型"].str.startswith("基本面")]) / len(exit_df) * 100, 1
            ) if len(exit_df) > 0 else 0
            if fund_pct > 0:
                st.info(f"基本面出場佔總交易 **{fund_pct}%**。若比例過高（>30%），代表技術進場點的基本面品質有改善空間；若比例極低（<5%），代表月營收篩選在進場端已過濾掉大部分弱勢股。")
            else:
                st.info("目前無基本面出場紀錄。確認 `update_data.py` 已執行完 Step 3（基本面歷史序列更新）。")
        else:
            st.info("出場原因欄位不存在，請重新在 Tab 3 執行一次回測以產生含基本面出場的數據。")

    # ==========================================
    # 模組 2：Robustness (參數高原熱力圖)
    # ==========================================
    st.subheader("2. 魯棒性檢定：參數高原熱力圖 (Parameter Surface Heatmap)")
    st.markdown("我們將針對**「移動停利 (RR)」**與**「最小缺口 (Gap%)」**進行 3x3 網格交叉測試 (Grid Search)。尋找一片綠色的「高原」，避免紅綠交錯的過度最佳化「孤峰」。")
    
    with st.expander("啟動參數高原壓力測試 (使用快取核心加速)"):
        st.warning("**運算警告：** 這將連續呼叫 9 次投資組合回測！因為我們已經換上快取(Cache)代理，速度會大幅提升，但仍建議先以【單一板塊】進行抽樣測試。")
        
        r_sec = st.multiselect("選擇抽樣板塊 (建議只選 1 個)", list(INDUSTRY_DICT.keys()), default=[list(INDUSTRY_DICT.keys())[0]], key="r_sec")
        
        grid_rr = st.text_input("輸入 3 個停利 RR 值 (用逗號隔開)", "8, 10, 12")
        grid_gap = st.text_input("輸入 3 個最小缺口 % (用逗號隔開)", "0.2, 0.3, 0.4")
        
        if st.button("啟動 3x3 網格熱力圖運算", type="primary"):
            rr_list = [float(x.strip()) for x in grid_rr.split(",")]
            gap_list = [float(x.strip()) for x in grid_gap.split(",")]
            
            if len(rr_list) != 3 or len(gap_list) != 3:
                st.error("請確保 RR 和 Gap 剛好都輸入 3 個數值！")
            else:
                heatmap_data = []
                r_tickers = []
                for sec in r_sec:
                    r_tickers.extend(INDUSTRY_DICT.get(sec, []))
                r_tickers = list(set(r_tickers))
                sector_map = {ticker: sec for sec, tickers in INDUSTRY_DICT.items() for ticker in tickers}
                
                progress_text = "正在進行多進程網格搜索 (Grid Search)..."
                my_bar = st.progress(0, text=progress_text)
                
                total_runs = len(gap_list) * len(rr_list)
                current_run = 0
                
                for g_val in gap_list:
                    for r_val in rr_list:
                        # 核心升級：呼叫快取版多進程引擎，飛速掃描！
                        cfg = PortfolioConfig(
                            tickers=r_tickers, years=10, bt_mode="移動停利", bt_val=r_val,
                            gap_pct=g_val, use_sma=True, use_vol_filter=False,
                            use_atr_filter=False, use_bos_filter=False,
                            slippage_pct=0.002, fee_disc=0.28, min_vol=1000,
                            max_concurrent=5, max_per_sector=2, sector_map=sector_map,
                            risk_pct=2.0, initial_capital=1000000.0, account_dd_limit=0.20,
                        )
                        h_df, h_stats, _ = get_cached_portfolio_results(cfg)
                        
                        if h_stats and not h_df.empty and h_stats.get("真實總報酬(%)", -100.0) != -100.0:
                            exp_val = h_stats["Portfolio期望值(R)"]
                        else:
                            exp_val = -1.0 
                            
                        heatmap_data.append({'Gap%': g_val, 'RR': r_val, '期望值(R)': exp_val})
                        
                        current_run += 1
                        my_bar.progress(current_run / total_runs, text=f"進度: {current_run}/{total_runs} (Gap={g_val}, RR={r_val})")
                
                my_bar.empty()
                
                hm_df = pd.DataFrame(heatmap_data)
                hm_pivot = hm_df.pivot(index='Gap%', columns='RR', values='期望值(R)')
                
                fig_hm = go.Figure(data=go.Heatmap(
                    z=hm_pivot.values,
                    x=hm_pivot.columns,
                    y=hm_pivot.index,
                    colorscale='RdYlGn', 
                    text=np.round(hm_pivot.values, 2),
                    texttemplate="%{text} R",
                    showscale=True
                ))
                
                fig_hm.update_layout(
                    title='SMC 策略參數高原熱力圖 (Parameter Surface)',
                    xaxis_title='停利 RR 倍數 (Profit Target)',
                    yaxis_title='最小缺口 Gap (%)',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_hm, use_container_width=True)
                
                st.info("**如何判讀熱力圖 (Robustness)：**\n如果你的 10 RR 與 0.3 Gap 是深綠色，但旁邊的 12 RR 卻突然變成深紅色，這代表你的策略是【過度最佳化】的孤峰，實盤必死。真正的聖杯應該是一整片顏色相近的**綠色高原**！")

    st.divider()

    # ==========================================
    # 模組 3：組合層級 Walk-Forward Optimization
    # ==========================================
    st.subheader("3. 組合 WFO 盲測驗證 (Portfolio Walk-Forward Optimization)")
    st.markdown(
        "這是最嚴格的驗證。系統會把歷史資料切成多個「訓練期 + 盲測期」，"
        "每次用訓練期找最佳參數，再拿去跑**從未見過的盲測期**。"
        "如果盲測期仍然有正期望值，才代表策略真的有 Alpha，而不是過擬合。"
    )

    if 'p_df' not in st.session_state or st.session_state['p_df'].empty:
        st.warning("請先至「投資組合 (Tab 3)」執行一次回測，產生標的清單後再來執行 WFO。")
    else:
        wfo_col1, wfo_col2, wfo_col3 = st.columns(3)
        with wfo_col1:
            wfo_train_years = st.number_input("訓練期（年）", min_value=2, max_value=5, value=3, step=1,
                help="每個視窗用幾年資料尋找最佳參數")
            wfo_test_years = st.number_input("盲測期（年）", min_value=1, max_value=2, value=1, step=1,
                help="每個視窗用幾年資料驗證（完全沒見過的未來）")
        with wfo_col2:
            wfo_gap_input = st.text_input("缺口 Gap 網格（逗號分隔）", "0.2, 0.3, 0.4",
                help="訓練期會在這幾個值裡搜尋最佳缺口")
            wfo_btval_input = st.text_input("停利 RR 網格（逗號分隔）", "8.0, 10.0, 12.0",
                help="訓練期會在這幾個值裡搜尋最佳停利")
            wfo_enable_purge = st.checkbox(
                "啟用資料洩漏防護 (Purging)", value=True,
                help="✅開啟(建議):訓練期只用『進出場都在訓練期內、且離盲測期有安全距離』的乾淨交易,"
                     "杜絕『持有期偷看盲測期』的資料洩漏。\n\n"
                     "❌關閉:舊行為(有洩漏),可用來對比洩漏對結果的影響。\n\n"
                     "注意:開啟後結果通常會更難看,因為擋掉了洩漏的『加持』——但那才是策略的真實表現。",
            )
        with wfo_col3:
            st.info(
                "**視窗數量：** 系統最多跑 6 個滾動視窗，"
                "每個視窗往前推一年，確保盲測涵蓋不同市場環境。\n\n"
                "**運算時間：** 每個視窗需跑多次組合回測，"
                "建議先用少數板塊測試。"
            )

        if st.button("啟動組合 WFO 盲測驗證", type="primary", use_container_width=True):
            try:
                wfo_gap_list   = [float(x.strip()) for x in wfo_gap_input.split(",")]
                wfo_btval_list = [float(x.strip()) for x in wfo_btval_input.split(",")]
            except ValueError:
                st.error("參數格式錯誤，請確認是數字並用逗號分隔。")
                st.stop()

            # 從 session_state 取得目前回測用的標的和設定
            wfo_tickers     = st.session_state.get('p_tickers', [])
            wfo_sector_map  = {ticker: sec for sec, tks in INDUSTRY_DICT.items() for ticker in tks}
            wfo_bt_mode     = st.session_state.get('p_bt_mode', '移動停利')
            wfo_risk        = st.session_state.get('p_risk', 2.0)
            wfo_capital     = st.session_state.get('p_cap', 1_000_000.0)
            wfo_dd_limit    = st.session_state.get('p_account_dd', 0.20)
            wfo_max_pos     = st.session_state.get('p_max_pos', 5)
            wfo_max_position = st.session_state.get('p_max_position', 0.15)  # 單股上限,與單次回測一致
            wfo_max_corr    = 0.75  # 相關性上限(單次回測預設值,一致)
            wfo_max_sec     = st.session_state.get('p_max_sec', 2)
            # ✅ 讀取 Tab3 設的進場過濾，確保 WFO 驗證的是同一個策略
            wfo_entry_yoy   = st.session_state.get('p_entry_min_yoy', None)
            wfo_entry_gm    = st.session_state.get('p_entry_require_gm', False)
            wfo_entry_ocf   = st.session_state.get('p_entry_require_ocf', False)
            wfo_regimes     = st.session_state.get('p_regimes', None)
            wfo_rotation    = st.session_state.get('p_rotation', False)
            wfo_rotation_n  = st.session_state.get('p_rotation_n_val', 3)
            wfo_full_map    = st.session_state.get('p_full_sector_map', None)

            if not wfo_tickers:
                st.error("找不到標的清單，請先在 Tab 3 執行回測。")
                st.stop()

            from portfolio_engine import run_portfolio_wfo

            total_combinations = len(wfo_gap_list) * len(wfo_btval_list)

            st.info(f"預計視窗數最多 8 個 × {total_combinations} 個參數組合。"
                    f"\n\n本地資料已預載，速度比之前快很多。")

            with st.status("WFO 執行中，請稍候...", expanded=True) as wfo_status_box:

                def _wfo_progress_cb(window_idx, total_windows, msg=""):
                    wfo_status_box.write(f"{'ok' if 'complete' in msg.lower() or '完成' in msg else '⏳'} {msg}")

                wfo_logs_df, wfo_stats = run_portfolio_wfo(
                    tickers=wfo_tickers,
                    bt_mode=wfo_bt_mode,
                    gap_options=wfo_gap_list,
                    bt_val_options=wfo_btval_list,
                    train_years=wfo_train_years,
                    test_years=wfo_test_years,
                    slippage_pct=0.002,
                    fee_disc=0.28,
                    min_vol=1000,
                    max_concurrent=wfo_max_pos,
                    max_per_sector=wfo_max_sec,
                    sector_map=wfo_sector_map,
                    risk_pct=wfo_risk,
                    initial_capital=wfo_capital,
                    account_dd_limit=wfo_dd_limit,
                    max_position_pct=wfo_max_position,
                    max_corr=wfo_max_corr,
                    entry_min_yoy=wfo_entry_yoy,
                    entry_require_gm=wfo_entry_gm,
                    entry_require_ocf=wfo_entry_ocf,
                    allowed_regimes=wfo_regimes,
                    enable_sector_rotation=wfo_rotation,
                    sector_top_n=wfo_rotation_n,
                    full_sector_map=wfo_full_map,
                    enable_purge=wfo_enable_purge,
                    progress_cb=_wfo_progress_cb,
                )

                wfo_status_box.update(label="WFO 完成！", state="complete", expanded=False)

            if wfo_logs_df.empty:
                st.error("WFO 未能產生任何結果，資料可能不足（需要至少 2015 年後的資料）。")
            else:
                # ── 綜合指標 ──
                if wfo_stats:
                    st.markdown("#### WFO 綜合盲測統計")
                    kc1, kc2, kc3, kc4 = st.columns(4)
                    kc1.metric("總盲測交易次數", f"{wfo_stats['總盲測交易次數']} 筆")
                    kc2.metric("WFO 綜合期望值", f"{wfo_stats['WFO綜合期望值(R)']} R",
                        delta="正期望值" if wfo_stats['WFO綜合期望值(R)'] > 0 else "負期望值 ❌")
                    kc3.metric("策略穩定性評分", f"{wfo_stats['策略穩定性評分']} %",
                        help="盲測期中正期望值視窗的比例，越高越穩定")
                    kc4.metric("正期望值視窗", wfo_stats['正期望值視窗比例'])

                    # 診斷訊息
                    score = wfo_stats['策略穩定性評分']
                    exp   = wfo_stats['WFO綜合期望值(R)']
                    if score >= 70 and exp > 0:
                        st.success(f"**策略通過 WFO 驗證！** 穩定性 {score}%，綜合期望值 {exp} R。在不同市場環境的盲測中仍有正期望值，Alpha 真實存在。")
                    elif score >= 50 and exp > 0:
                        st.warning(f"**策略部分通過。** 穩定性 {score}%，有些年份盲測是負的。建議縮小部位或繼續觀察。")
                    else:
                        st.error(f"**策略未通過 WFO 驗證。** 穩定性只有 {score}%，盲測期望值 {exp} R。很可能是過擬合，不建議實盤。")

                # ── 各視窗明細表 ──
                st.markdown("#### 各盲測視窗明細")

                # 顏色標記：OOS 期望值正 → 綠，負 → 紅
                def _color_oos(val):
                    if isinstance(val, (int, float)):
                        return "color: #00e676" if val > 0 else "color: #ff5252"
                    return ""

                st.dataframe(
                    wfo_logs_df.style.applymap(_color_oos, subset=["OOS期望值(R)", "OOS淨利(R)"]),
                    use_container_width=True,
                    hide_index=True,
                )

                # ── OOS 期望值視覺化 ──
                fig_wfo = go.Figure()
                fig_wfo.add_trace(go.Bar(
                    x=wfo_logs_df["盲測期間"],
                    y=wfo_logs_df["OOS期望值(R)"],
                    marker_color=["#00e676" if v > 0 else "#ff5252" for v in wfo_logs_df["OOS期望值(R)"]],
                    name="OOS 期望值 (R)",
                ))
                fig_wfo.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                fig_wfo.update_layout(
                    template="plotly_dark",
                    title="各盲測視窗 OOS 期望值（綠=正，紅=負）",
                    xaxis_title="盲測期間",
                    yaxis_title="期望值 (R)",
                )
                st.plotly_chart(fig_wfo, use_container_width=True)

                st.caption(
                    "判讀方式：每個柱子代表一個盲測期。"
                    "綠色代表該期間策略有正期望值，紅色代表虧損。"
                    "如果大多數是綠色，代表策略在不同時期都能賺錢，Alpha 真實；"
                    "如果紅綠交錯或大多數是紅色，代表策略依賴特定時期，過擬合風險高。"
                )

    st.divider()

    # ==========================================
    # 模組 4：月營收雷達回溯驗證
    # ==========================================
    st.subheader("4. 月營收雷達回溯驗證（Radar Backtest）")
    st.markdown(
        "驗證月營收雷達**在每個歷史時間點實際會選出哪些板塊**，"
        "確認它在 2019 年選到半導體、2021 年選到 AI 伺服器，"
        "而不是永遠慢半拍。\n\n"
        "需要先執行 `python update_data.py` 完成 Step 3（歷史基本面序列下載）才有資料。"
    )

    with st.expander("啟動雷達回溯驗證"):
        rb_col1, rb_col2, rb_col3 = st.columns(3)
        rb_start   = rb_col1.number_input("起始年份", min_value=2010, max_value=2023, value=2016, step=1)
        rb_top_n   = rb_col2.number_input("每次選幾個板塊", min_value=1, max_value=10, value=3, step=1)
        rb_min_yoy = rb_col3.number_input("YoY 門檻 (%)", min_value=0.0, max_value=50.0, value=10.0, step=5.0)
        rb_freq    = st.radio("驗證頻率", ["每季", "每年"], horizontal=True)

        # ── 快取函數，參數沒變就不重算 ──
        @st.cache_data(ttl=3600, show_spinner=False)
        def _load_fund_hist_cached(tickers_tuple):
            """只讀本地 parquet，不打 FinMind API"""
            import os
            from revenue_fetcher import REVENUE_HISTORY_DIR, MARGIN_HISTORY_DIR
            cache = {}
            for t in tickers_tuple:
                ticker_pure = str(t).replace(".TW", "")
                rev_path    = os.path.join(REVENUE_HISTORY_DIR, f"{ticker_pure}.parquet")
                margin_path = os.path.join(MARGIN_HISTORY_DIR,  f"{ticker_pure}.parquet")
                rev    = pd.read_parquet(rev_path)    if os.path.exists(rev_path)    else pd.DataFrame()
                margin = pd.read_parquet(margin_path) if os.path.exists(margin_path) else pd.DataFrame()
                cache[t] = {"rev": rev, "margin": margin}
            return cache

        @st.cache_data(ttl=3600, show_spinner=False)
        def _load_price_hist_cached(tickers_tuple):
            """只讀本地 parquet K 線，只保留 Close 欄位"""
            import os
            from data_provider import DATA_DIR
            hist = {}
            for t in tickers_tuple:
                stock_id = str(t).replace(".TW.TW", "").replace(".TW", "")
                path = os.path.join(DATA_DIR, f"{stock_id}.TW.parquet")
                if os.path.exists(path):
                    try:
                        df = pd.read_parquet(path, columns=["Close"])
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        hist[t] = df
                    except Exception:
                        pass
            return hist

        @st.cache_data(ttl=3600, show_spinner=False)
        def _run_radar_backtest_cached(sector_map_key, start_year, top_n, min_yoy, freq,
                                        tickers_tuple):
            """整個回溯計算快取，參數一樣就直接回傳上次結果"""
            from sector_radar import SectorRadar
            fund_hist  = _load_fund_hist_cached(tickers_tuple)
            price_hist = _load_price_hist_cached(tickers_tuple)
            radar = SectorRadar(SECTOR_MAP)
            return radar.backtest_radar_selections(
                fundamental_history=fund_hist,
                start_year=start_year,
                top_n=top_n,
                min_yoy=min_yoy,
                freq="Q" if freq == "每季" else "Y",
                price_history=price_hist,
            )

        if st.button("執行雷達回溯驗證", type="primary", use_container_width=True):
            import os
            from revenue_fetcher import REVENUE_HISTORY_DIR
            if not os.path.exists(REVENUE_HISTORY_DIR) or len(os.listdir(REVENUE_HISTORY_DIR)) == 0:
                st.error("找不到歷史月營收資料。請先執行 `python update_data.py` 完成 Step 3。")
                st.stop()

            all_tickers = tuple(sorted({t for tks in SECTOR_MAP.values() for t in tks}))

            prog1 = st.empty()
            prog1.info("Step 1/3：載入歷史月營收序列（讀本地快取）...")
            prog1.empty()

            prog2 = st.empty()
            prog2.info("Step 2/3：載入歷史 K 線（讀本地快取）...")
            prog2.empty()

            with st.spinner("Step 3/3：計算各季度複合板塊分數..."):
                radar_df = _run_radar_backtest_cached(
                    sector_map_key=str(sorted(SECTOR_MAP.keys())),
                    start_year=int(rb_start),
                    top_n=int(rb_top_n),
                    min_yoy=float(rb_min_yoy),
                    freq=rb_freq,
                    tickers_tuple=all_tickers,
                )

            if radar_df.empty:
                st.warning("回溯驗證無結果，確認歷史資料是否完整。")
            else:
                # ── 各時間點選出的板塊 ──
                st.markdown("#### 各時間點雷達選出的板塊")
                display_df = radar_df[["date", "top_sectors", "data_coverage"]].copy()
                display_df["top_sectors"] = display_df["top_sectors"].apply(
                    lambda x: "、".join(x) if x else "無資料"
                )
                display_df.columns = ["時間點", "雷達選出的板塊", "有資料的板塊數"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # ── 板塊被選中次數排行 ──
                st.markdown("#### 各板塊被雷達選中的次數")
                sector_counts = {}
                for sectors in radar_df["top_sectors"]:
                    for s in sectors:
                        sector_counts[s] = sector_counts.get(s, 0) + 1

                if sector_counts:
                    count_df = pd.DataFrame(
                        sorted(sector_counts.items(), key=lambda x: x[1], reverse=True),
                        columns=["板塊", "被選中次數"]
                    )
                    total_periods = len(radar_df)
                    count_df["被選中比例"] = (count_df["被選中次數"] / total_periods * 100).round(1).astype(str) + "%"

                    fig_radar_bt = go.Figure(go.Bar(
                        x=count_df["板塊"],
                        y=count_df["被選中次數"],
                        marker_color=["#00e676" if v >= total_periods * 0.5 else "#448aff"
                                      if v >= total_periods * 0.25 else "#78909c"
                                      for v in count_df["被選中次數"]],
                        text=count_df["被選中比例"],
                        textposition="outside",
                    ))
                    fig_radar_bt.update_layout(
                        template="plotly_dark",
                        title=f"板塊被雷達選中次數（共 {total_periods} 個時間點）",
                        xaxis_title="板塊",
                        yaxis_title="被選中次數",
                        xaxis_tickangle=-45,
                    )
                    st.plotly_chart(fig_radar_bt, use_container_width=True)

                    st.caption(
                        "**判讀方式：** 綠色 = 超過一半時間都被選中（核心強勢板塊）；"
                        "藍色 = 偶爾被選中（輪動板塊）；灰色 = 少數時期才出現。"
                        "如果半導體和電子在 2019-2022 年是綠色，代表雷達能動態捕捉到那段多頭；"
                        "如果任何時期都是同一個板塊，代表雷達沒有真正在輪動，需要調整評分邏輯。"
                    )
# ─── Tab 6：實盤對帳 ───────────────────────────────────────────────────────
with tab5:
    st.header("實盤對帳系統")
    st.caption("記錄每筆實際成交，與系統訊號比對，追蹤滑價與 R 值差距。")
    from trade_journal import (
        log_signal_entry, confirm_entry, confirm_exit,
        load_live_trades, get_slippage_summary
    )

    j_tab1, j_tab2 = st.tabs(["交易記錄", "新增 / 更新"])

    with j_tab1:
        live_df = load_live_trades()
        if not live_df.empty:
            # 滑價彙總
            summary = get_slippage_summary()
            if summary:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("平均進場滑價", f"{summary['avg_entry_slippage_pct']} %")
                s2.metric("平均出場滑價", f"{summary['avg_exit_slippage_pct']} %")
                s3.metric("系統平均 R",   f"{summary['avg_signal_r']} R")
                s4.metric("實際平均 R",   f"{summary['avg_actual_r']} R",
                          delta=f"衰減 {summary['r_decay']} R",
                          delta_color="inverse" if summary['r_decay'] > 0 else "normal")
            st.divider()
            status_filter = st.selectbox("篩選狀態", ["all", "open", "closed", "pending", "cancelled"])
            show_df = load_live_trades(None if status_filter == "all" else status_filter)
            st.dataframe(show_df, use_container_width=True, hide_index=True)
        else:
            st.info("尚無實盤記錄。在右側「新增 / 更新」頁籤輸入第一筆。")

    with j_tab2:
        action = st.radio("操作", ["記錄新訊號", "確認進場", "確認出場"], horizontal=True)

        if action == "記錄新訊號":
            with st.form("new_signal"):
                c1, c2, c3 = st.columns(3)
                j_ticker   = c1.text_input("標的代號", placeholder="2330.TW")
                j_sector   = c2.text_input("板塊", placeholder="半導體業")
                j_date     = c3.date_input("訊號日期")
                c4, c5, c6 = st.columns(3)
                j_sig_price = c4.number_input("訊號進場價", min_value=0.0, step=0.5)
                j_sig_stop  = c5.number_input("訊號止損價", min_value=0.0, step=0.5)
                j_shares    = c6.number_input("股數（張×1000）", min_value=0, step=1000)
                j_notes     = st.text_input("備註")
                if st.form_submit_button("記錄訊號", type="primary"):
                    ok = log_signal_entry(j_ticker, j_sector, j_date.isoformat(),
                                          j_sig_price, j_sig_stop, j_shares, j_notes)
                    st.success("訊號已記錄") if ok else st.error("記錄失敗")

        elif action == "確認進場":
            with st.form("confirm_entry"):
                j_id          = st.number_input("交易 ID（從記錄表查看）", min_value=1, step=1)
                c1, c2        = st.columns(2)
                j_actual_entry = c1.number_input("實際成交價", min_value=0.0, step=0.5)
                j_actual_stop  = c2.number_input("實際止損掛單價", min_value=0.0, step=0.5)
                j_entry_date   = st.date_input("實際進場日期")
                if st.form_submit_button("確認進場", type="primary"):
                    ok = confirm_entry(int(j_id), j_actual_entry, j_actual_stop, j_entry_date.isoformat())
                    st.success("進場已確認") if ok else st.error("更新失敗，確認 ID 是否正確")

        else:
            with st.form("confirm_exit"):
                j_id_ex       = st.number_input("交易 ID", min_value=1, step=1)
                c1, c2        = st.columns(2)
                j_actual_exit  = c1.number_input("實際出場價", min_value=0.0, step=0.5)
                j_exit_reason  = c2.selectbox("出場原因", ["Hit_Stop", "Hit_Target", "移動停利", "基本面出場", "手動出場", "其他"])
                j_exit_date    = st.date_input("實際出場日期")
                if st.form_submit_button("確認出場", type="primary"):
                    ok = confirm_exit(int(j_id_ex), j_actual_exit, j_exit_reason, j_exit_date.isoformat())
                    st.success("出場已確認") if ok else st.error("更新失敗，確認 ID 是否正確")


# ─── Tab 7：參數變更日誌 ───────────────────────────────────────────────────
with tab6:
    st.header("參數變更日誌")
    st.caption("每次調整策略參數都應記錄，方便事後追查績效變化的原因。")
    from trade_journal import log_param_change, load_param_changelog

    with st.expander("記錄本次參數變更"):
        with st.form("param_log"):
            p1, p2, p3 = st.columns(3)
            pl_name    = p1.text_input("參數名稱", placeholder="min_yoy / bt_val / risk_pct ...")
            pl_old     = p2.text_input("變更前數值")
            pl_new     = p3.text_input("變更後數值")
            pl_reason  = st.text_area("變更原因（簡短說明）",
                placeholder="例：WFO 顯示 2024 表現弱，提高 YoY 門檻從 15% 到 20% 以提升選股品質")
            if st.form_submit_button("記錄變更", type="primary"):
                ok = log_param_change(pl_name, pl_old, pl_new, pl_reason)
                st.success("已記錄") if ok else st.error("記錄失敗")

    st.divider()
    changelog_df = load_param_changelog()
    if not changelog_df.empty:
        st.dataframe(changelog_df, use_container_width=True, hide_index=True)
    else:
        st.info("尚無參數變更記錄。")