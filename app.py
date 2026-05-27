import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import logging
import requests
import datetime

# 🌟 網頁設定拉到最頂端
st.set_page_config(page_title="SMC 量化拷問終端 V19.0", layout="wide")

# --- 新增工具函數 ---
def safe_divide(a, b):
    """避免除以零錯誤的安全除法"""
    try:
        return a / b
    except ZeroDivisionError:
        return 0

def send_telegram_async(msg):
    """非同步發送 Telegram 訊息 (啟用額外執行緒)"""
    import threading
    def _send():
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': msg,
            'parse_mode': 'Markdown'
        }
        try:
            requests.post(url, json=payload)
        except Exception as e:
            logger.warning(f"Telegram 發送失敗: {e}")
    thread = threading.Thread(target=_send, daemon=True)
    thread.start()

# 初始化 Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_provider import load_data, fetch_stock_data
from strategy import apply_smc_logic
from backtest_engine import run_backtest_core
from database import init_db, save_backtest_record, load_history, save_backtest_records_batch
from monte_carlo import run_monte_carlo
from portfolio_engine import run_portfolio_backtest

# 🌟 核心升級：精準對齊 kwargs，讓選配參數安全送達底層
@st.cache_data(ttl=3600)
def get_cached_portfolio_results(tickers, years, mode, val, gap, sma, vol, atr, bos, slip, fee, minv, maxp, maxs, smap, risk, cap, account_dd):
    return run_portfolio_backtest(
        tickers=tickers, years=years, bt_mode=mode, bt_val=val, gap_pct=gap, use_sma=sma,
        use_vol_filter=vol, use_atr_filter=atr, use_bos_filter=bos,  # 精準對應 strategy.py 的命名
        slippage_pct=slip, fee_disc=fee, min_vol=minv, 
        max_concurrent=maxp, max_per_sector=maxs, sector_map=smap, 
        risk_pct=risk, initial_capital=cap, account_dd_limit=account_dd
    )

# 初始化資料庫
init_db()
INDUSTRY_DICT, STOCK_NAMES, SEARCH_LIST = load_data()

st.title("SMC 台股科學選股引擎 V19.0 (多核運算版)")
st.caption("Tier 5 機構級驗證：動態部位縮減、移動停利解鎖、蒙地卡羅壓力測試")

# --- Telegram 配置 ---
TELEGRAM_BOT_TOKEN ='8902281081:AAGSkks37g_wcpcf5KEulXsthf_jkin9hy0'
TELEGRAM_CHAT_ID ='7779789363'

# 分頁宣告
tab1, tab2, tab3, tab4, tab5 = st.tabs(["即時掃描雷達", "深度回測實驗室", "投資組合回測 (Portfolio)", "歷史戰績資料庫", "防禦與衰減檢定"])

# ==========================================
# 🌟 分頁 1：即時掃描雷達
# ==========================================
with tab1:
    st.header("SMC 動能掃描雷達")
    st.caption("同步回測邏輯：導入機構級部位派單與 SMC 動態濾網")
    
    INDUSTRY_DICT_REVERSE = {ticker: sec for sec, tickers in INDUSTRY_DICT.items() for ticker in tickers}
    
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
        r_sectors = st.multiselect("掃描板塊", list(INDUSTRY_DICT.keys()), default=list(INDUSTRY_DICT.keys())[:3])

    if st.button("執行全自動動能掃描與派單", use_container_width=True, type="primary"):
        target_tickers = []
        for sec in r_sectors:
            target_tickers.extend(INDUSTRY_DICT.get(sec, []))
            
        with st.spinner("正在檢查大盤宏觀環境..."):
            m_data = fetch_stock_data("0050.TW", days=100)
            if m_data is not None and not m_data.empty:
                m_close = m_data['Close'].iloc[-1]
                m_sma_val = m_data['Close'].rolling(60).mean().iloc[-1]
                m_change = (m_close / m_data['Close'].iloc[-2] - 1) * 100

                if m_close < m_sma_val:
                    st.error("大盤斷路器啟動：0050 位於季線下方，目前環境極度危險，系統拒絕發出任何買入建議。")
                    st.stop()
                if m_change < -2.0:
                    st.warning(f"市場恐慌警告：今日大盤跌幅達 {m_change:.2f}%，建議放棄所有新掛單。")

        radar_results = []
        progress_bar = st.progress(0)
        
        with st.spinner(f"正在掃描 {len(target_tickers)} 檔標的..."):
            for i, ticker in enumerate(target_tickers):
                progress_bar.progress((i + 1) / len(target_tickers))
                try:
                    df = fetch_stock_data(ticker, days=100)
                    if df is None or df.empty or len(df) < 65:
                        continue
                        
                    # 🌟 參數同步：將雷達 UI 設定傳給核心大腦
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
                        vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
                        vol_surge = safe_divide(df['Volume'].iloc[-1], vol_ma20)
                        entry_price = last_row['Low']
                        sl_price = entry_price - last_row['adjusted_risk']
                        tp_price = entry_price * 1.10
                        rr_ratio = safe_divide((tp_price - entry_price), last_row['adjusted_risk'])
                        
                        if vol_surge >= min_momentum and rr_ratio >= min_rr:
                            risk_amount = live_equity * live_risk_pct
                            risk_per_share = max(entry_price - sl_price, 0.01) 
                            recommended_shares = int(risk_amount / risk_per_share)
                            
                            capital_required = recommended_shares * entry_price
                            max_cash_allowed = live_equity * max_capital_per_trade
                            
                            if capital_required > max_cash_allowed:
                                recommended_shares = int(max_cash_allowed / entry_price)
                                capital_required = recommended_shares * entry_price
                            
                            if recommended_shares <= 0:
                                continue

                            radar_results.append({
                                "股票代碼": ticker,
                                "板塊": INDUSTRY_DICT_REVERSE.get(ticker, "其他"),
                                "動能分數": round(vol_surge, 2),
                                "預期賠率(R)": round(rr_ratio, 1),
                                "進場價": round(entry_price, 2),
                                "防彈止損價": round(sl_price, 2),
                                "獲利目標": round(tp_price, 2),
                                "建議買入股數": recommended_shares,
                                "預估耗用資金": round(capital_required, 2)
                            })
                            
                            tg_msg = (
                                f"*[SMC 機構派單中心] 鎖定獵物！*\n"
                                f"*標的*：{ticker} ({INDUSTRY_DICT_REVERSE.get(ticker, '其他')})\n"
                                f"*動能*：{vol_surge:.2f}x | *賠率*：{rr_ratio:.1f} R\n"
                                f"➖➖➖➖➖➖➖➖\n"
                                f"*進場掛單*：`{entry_price:.2f}`\n"
                                f"*防彈止損*：`{sl_price:.2f}`\n"
                                f"*獲利目標*：`{tp_price:.2f}`\n"
                                f"*建議買入股數*：`{recommended_shares}` 股\n"
                                f"*預估耗用資金*：${capital_required:,.0f}"
                            )
                            send_telegram_async(tg_msg)
                            
                except Exception as e:
                    continue

        progress_bar.empty()
        
        if radar_results:
            radar_df = pd.DataFrame(radar_results).sort_values("動能分數", ascending=False)
            st.success(f"掃描完成！發現 {len(radar_df)} 檔完美目標。")
            
            st.subheader("今日實盤派單清單 (依聰明錢力道排序)")
            styled_df = radar_df.style.format({
                '動能分數': '{:.2f}', '預期賠率(R)': '{:.2f} R',
                '進場價': '${:.2f}', '防彈止損價': '${:.2f}', '獲利目標': '${:.2f}',
                '建議買入股數': '{:,.0f} 股', '預估耗用資金': '${:,.0f}'
            }).map(lambda x: "background-color: #004d40; color: white;", subset=['動能分數', '建議買入股數'])
            
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
            st.info("操作建議：請直接打開券商 App，依照「建議買入股數」與「進場掛單價」進行限價掛單。")
        else:
            st.warning(f"今日收盤無符合嚴格條件的標的。今日空手，享受生活保護資金！")

# ==========================================
# 🌟 分頁 2：單股深度回測實驗室
# ==========================================
with tab2:
    st.header("科學化回測實驗室 (單股)")
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
        
        df_is_res, t_is, w_is, exp_is, mdd_is, sharpe_is = run_backtest_core(df_is, bt_mode, bt_val, slippage_pct, fee_disc, 0.003, min_vol)
        df_oos_res, t_oos, w_oos, exp_oos, mdd_oos, sharpe_oos = run_backtest_core(df_oos, bt_mode, bt_val, slippage_pct, fee_disc, 0.003, min_vol)
        
        st.markdown(f"### OOS 盲測對比")
        c_is, c_oos = st.columns(2)
        with c_is:
            st.success(f"**訓練期 (IS)**: {exp_is:.2f} R")
            st.write(f"夏普值: {sharpe_is:.2f}")
        with c_oos:
            st.warning(f"**盲測期 (OOS)**: {exp_oos:.2f} R")
            st.write(f"夏普值: {sharpe_oos:.2f}")

        st.markdown("### 盲測期 (OOS) 真實交易明細")
        if not df_oos_res.empty:
            st.dataframe(df_oos_res, use_container_width=True, hide_index=True)
        else:
            st.info("盲測期間沒有觸發任何交易。")

        st.divider()
        st.subheader("蒙地卡羅壓力測試 (1,000 次序列重抽)")
        mc_results = run_monte_carlo(df_oos_res) 
        
        if mc_results:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("中位數預期獲利", f"{mc_results['median_final_r']:.2f} R")
            m2.metric("破產機率 (DD > 30%)", f"{mc_results['prob_of_ruin_pct']:.1f}%", delta_color="inverse")
            m3.metric("中位數最大回撤", f"{mc_results['median_max_dd_r']:.2f} R")
            m4.metric("極端最慘回撤", f"{mc_results['worst_max_dd_r']:.2f} R")
        else:
            st.info("交易次數不足，無法執行蒙地卡羅模擬。")

# ==========================================
# 🌟 分頁 3：投資組合 Alpha 驗證引擎
# ==========================================
with tab3:
    st.header("投資組合 Alpha 驗證引擎 (機構級)")
    
    p_col1, p_col2 = st.columns([1, 4])
    with p_col1:
        p_sector = st.multiselect("測試板塊", list(INDUSTRY_DICT.keys()), default=[list(INDUSTRY_DICT.keys())[0]])
        p_years = st.slider("回測年限", 1, 15, 10, key="py_slider")
        p_cap = st.number_input("初始本金", value=1000000, step=100000, key="pcap_input")
        p_risk = st.slider("單筆風險 (%)", 0.5, 5.0, 2.0, 0.5, key="prisk_slider")
        p_max_pos = st.slider("最大持倉數", 1, 20, 5)
        p_max_sec = st.slider("單一產業上限", 1, 5, 2)
        p_account_dd = st.slider("帳戶停機線 (%)", 10, 50, 20, 5) / 100.0
        
        st.divider()
        p_mode = st.selectbox("出場方式", ["固定盈虧比", "移動停利"], key="pmode_sel")
        p_val = st.number_input("目標倍數/停利%", value=10.0, key="pval_num")
        p_gap = st.number_input("最小缺口(%)", value=0.0, step=0.1, key="pgap_num")
        p_sma = st.checkbox("季線濾網", value=True, key="psma_chk")
        
        st.markdown("###### 🎛️ SMC 嚴格選配模組")
        p_vol = st.checkbox("要求起漲爆量", value=False, key="pvol_chk")
        p_atr = st.checkbox("強勢實體位移", value=False, key="patr_chk")
        p_bos = st.checkbox("突破近期新高", value=False, key="pbos_chk")
        
        run_portfolio = st.button("🚀 執行組合驗證", type="primary", use_container_width=True)

    with p_col2:
        if run_portfolio:
            if not p_sector:
                st.warning("請至少選擇一個板塊！")
            else:
                tickers_to_test = []
                for sec in p_sector:
                    tickers_to_test.extend(INDUSTRY_DICT.get(sec, []))
                tickers_to_test = list(set(tickers_to_test))
                sector_map = {ticker: sec for sec, tickers in INDUSTRY_DICT.items() for ticker in tickers}
                
                with st.spinner(f"正在平行運算 {len(tickers_to_test)} 檔標的，請稍候..."):
                    p_df, p_stats, bm_df = get_cached_portfolio_results(
                        tickers_to_test, p_years, p_mode, p_val, p_gap, p_sma,
                        p_vol, p_atr, p_bos, 0.002, 0.28, 1000, 
                        p_max_pos, p_max_sec, sector_map, p_risk, p_cap, p_account_dd
                    )
                    
                    if p_stats and not p_df.empty:
                        st.session_state['p_df'] = p_df
                        st.session_state['p_stats'] = p_stats
                        st.session_state['bm_df'] = bm_df
                        st.session_state['p_cap'] = p_cap
                    else:
                        st.error("❌ 引擎未觸發交易。請嘗試調低「最小缺口」或取消勾選「嚴格選配模組」。")

    # ==========================================
    # 🌟 將報表呈現區移出側邊欄，享受全螢幕寬度！
    # ==========================================
    if 'p_df' in st.session_state and not st.session_state['p_df'].empty:
        p_df = st.session_state['p_df']
        p_stats = st.session_state['p_stats']
        bm_df = st.session_state.get('bm_df', pd.DataFrame()) # 補上遺失的 benchmark 讀取
        p_cap_state = st.session_state.get('p_cap', 1000000)  # 修正 NameError：安全提取本金
        
        st.divider()
        st.markdown(f"### 📈 投資組合綜合績效報表 (初始本金：NT$ {p_cap_state:,})")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("期末總淨值", f"NT$ {p_stats['期末總淨值(NTD)']:,}")
        m2.metric("最大回撤", f"-{p_stats['真實最大回撤(%)']}%")
        m3.metric("總成交", f"{p_stats['總交易次數']} 次")
        m4.metric("淘汰次數", f"{p_stats['風控與資金不足淘汰次數']} 次")
        m5.metric("期望值", f"{p_stats['Portfolio期望值(R)']} R")
        
        st.subheader(f"資金曲線對決：SMC 策略 vs 0050 大盤")
        fig_p = go.Figure()
        
        fig_p.add_trace(go.Scatter(x=p_df['出場日期'], y=p_df['交易後總淨值'], mode='lines', name='SMC 策略 (真資金)', line=dict(color='cyan', width=2)))
        
        if not bm_df.empty:
            date_col = 'Date' if 'Date' in bm_df.columns else 'index' if 'index' in bm_df.columns else bm_df.columns[0]
            # 修正 NameError
            bm_equity = p_cap_state * (1 + bm_df['基準報酬(%)'] / 100) 
            bm_final = round(bm_equity.iloc[-1], 0)
            fig_p.add_trace(go.Scatter(x=bm_df[date_col], y=bm_equity, mode='lines', name=f'0050 持有 (NT$ {bm_final:,.0f})', line=dict(color='yellow', width=2, dash='dot')))
        
        fig_p.update_layout(template="plotly_dark", hovermode="x unified", height=500)
        st.plotly_chart(fig_p, use_container_width=True)
        
        st.divider()
        st.markdown("### 🕵️‍♂️ 避險基金級深度診斷 (Hedge Fund Analytics)")
        
        fig_dd = px.area(p_df, x='出場日期', y='真實回撤(%)', title='水下回撤曲線 (Underwater Curve)', height=300)
        fig_dd.update_traces(fillcolor='rgba(255, 0, 0, 0.3)', line=dict(color='red', width=1))
        fig_dd.update_layout(template="plotly_dark", yaxis_title='距離歷史高點回撤 (%)', hovermode="x unified")
        st.plotly_chart(fig_dd, use_container_width=True)
        
        st.markdown("#### 贏家集中度與年度拆解")
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

        with st.expander("🔍 查看這 5 筆極端值交易明細"):
            st.dataframe(top_5_trades[['日期', '出場日期', '標的', '板塊', 'Regime', '真實R', '絕對損益']], hide_index=True)

        st.divider()
        st.markdown("#### 🎲 蒙地卡羅破產模擬 (真・複利版)")
        st.caption("將歷史交易順序打亂重抽，並以真實複利(Compounding)計算淨值，測試連虧地獄。")
        
        trade_results = p_df['真實R'].tolist()
        
        if len(trade_results) < 20:
            st.warning("⚠️ 實際交易樣本數低於 20 筆，執行蒙地卡羅模擬易產生「統計幻覺」，已自動攔截。建議放寬策略參數。")
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
                sim_equity[0] = p_cap_state # 修正 NameError
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
                    # 修正 NameError
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
with tab4:
    st.header("歷史戰績資料庫")
    history_df = load_history()
    if not history_df.empty: st.dataframe(history_df, use_container_width=True, hide_index=True)
    else: st.info("目前尚無存檔。")

# --- 分頁 5：終極防禦與衰減檢定 (Robustness & Alpha Decay) ---
with tab5:
    st.header("終極防禦與衰減檢定 (Hedge Fund Due Diligence)")
    st.caption("驗證參數是否過度最佳化 (Overfitting) 以及 Alpha 優勢是否隨時間衰減。")
    
    st.divider()
    
    # ==========================================
    # 模組 1：Alpha Decay (前向滾動時間衰減檢定)
    # ==========================================
    st.subheader("1. Alpha 衰減檢定 (Walk-Forward Alpha Decay)")
    st.markdown("真正的優勢(Alpha)不該像冰塊一樣快速融化。我們將檢視策略在過去 10 年間，每年的**期望值(R)**與**勝率**是否保持平穩。")
    
    # 🌟 透過 session_state 讀取資料，不再受限於 Tab 切換重整問題！
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
                        # 🌟 核心升級：呼叫快取版多進程引擎，飛速掃描！
                        h_df, h_stats, _ = get_cached_portfolio_results(
                            tickers=r_tickers, years=10, bt_mode="移動停利", bt_val=r_val, gap_pct=g_val, 
                            use_sma=True, slippage_pct=0.1, fee_disc=0.28, min_vol=1000, 
                            max_concurrent=5, max_per_sector=2, sector_map=sector_map, 
                            risk_pct=0.02, initial_capital=1000000.0
                        )
                        
                        if h_stats and not h_df.empty and h_stats["真實總報酬(%)"] != -100.0:
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
                
                st.info("💡 **如何判讀熱力圖 (Robustness)：**\n如果你的 10 RR 與 0.3 Gap 是深綠色，但旁邊的 12 RR 卻突然變成深紅色，這代表你的策略是【過度最佳化】的孤峰，實盤必死。真正的聖杯應該是一整片顏色相近的**綠色高原**！")
                #streamlit run app.py