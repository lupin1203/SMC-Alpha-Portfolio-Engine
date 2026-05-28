# portfolio_engine.py
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import pandas as pd
import numpy as np
import logging
import traceback
from strategy import apply_smc_logic
from backtest_engine import run_backtest_core
from data_provider import fetch_stock_data

# 🌟 透過 **kwargs 無腦接收所有 UI 參數
def _pure_calc_worker(ticker, df_raw, **kwargs):
    try:
        # 0. 基礎資料清潔
        df_raw = df_raw.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
        if len(df_raw) < 60: return None

        # 計算動能分數
        df_raw['Vol_MA20'] = df_raw['Volume'].rolling(20).mean()
        df_raw['Score_VolSurge'] = df_raw['Volume'] / df_raw['Vol_MA20']
        
        # 從 kwargs 中提取策略參數
        gap_pct = kwargs.get('gap_pct', 0.3)
        use_sma = kwargs.get('use_sma', True)
        
        # 呼叫策略邏輯
        df_logic = apply_smc_logic(
            df_raw, 
            min_gap_pct=gap_pct, 
            use_sma_filter=use_sma,
            **kwargs 
        )
        
        # 執行回測
        res = run_backtest_core(
            df=df_logic, 
            bt_mode=kwargs.get('bt_mode', 'R倍數'), 
            bt_val=kwargs.get('bt_val', 3.0), 
            slippage_pct=kwargs.get('slippage_pct', 0.002), 
            fee_disc=kwargs.get('fee_disc', 0.28),
            tax_rate=0.003,       
            min_vol=kwargs.get('min_vol', 1000)
        )
        
        if res[0] is not None and not res[0].empty:
            trades_df = res[0].copy()
            trades_df['標的'] = ticker
            trades_df['板塊'] = kwargs.get('sector_map', {}).get(ticker, 'Unknown')
            
            # 🌟 審計優化：使用精準的 .loc 映射，杜絕 get_indexer 的 -1 錯位問題
            entry_dates = pd.to_datetime(trades_df['日期'])
            
            if 'Score_VolSurge' in df_raw.columns:
                trades_df['Alpha動能分數'] = df_raw.loc[entry_dates, 'Score_VolSurge'].values
            else:
                trades_df['Alpha動能分數'] = 1.0
                
            trades_df['Volume'] = df_raw.loc[entry_dates, 'Volume'].values 
            
            if '每股風險' not in trades_df.columns:
                trades_df['每股風險'] = trades_df['進場'] * 0.05
                
            return trades_df
            
    except Exception as e:
        logging.error(f"單股回測崩潰 ({ticker}): {e}")
        logging.error(traceback.format_exc())
    return None

def run_portfolio_backtest(tickers, years, bt_mode, bt_val, gap_pct, use_sma, slippage_pct, fee_disc, min_vol, max_concurrent=5, max_per_sector=2, sector_map=None, risk_pct=2.0, initial_capital=1000000.0, account_dd_limit=0.20, **kwargs):
    
    print(f"\n==========================================")
    print(f"⚙️ [組合回測引擎] 啟動！接收 {len(tickers)} 檔標的。")
    print(f"==========================================")

    risk_decimal = risk_pct / 100.0 
    raw_data_dict = {}
    all_trades = []
    
    def _fetch_worker(t):
        df = fetch_stock_data(t, days=years * 365)
        if df is not None and not df.empty and len(df) >= 100:
            return t, df
        return t, None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_worker, t): t for t in tickers}
        for future in as_completed(futures):
            t, df = future.result()
            if df is not None:
                raw_data_dict[t] = df

    if not raw_data_dict:
        print("❌ 錯誤：資料抓取失敗或全數為空。")
        return pd.DataFrame(), None, pd.DataFrame()

    market_df = fetch_stock_data("0050.TW", days=years*365)
    if market_df is not None:
        market_df['SMA60'] = market_df['Close'].rolling(60).mean()
        market_df['SMA200'] = market_df['Close'].rolling(200).mean()
        market_df['Daily_Return'] = market_df['Close'].pct_change()
        market_df['Volatility_20d'] = market_df['Daily_Return'].rolling(20).std() * np.sqrt(252) * 100
        market_df['Regime'] = np.where(market_df['Close'] > market_df['SMA200'], 'Bull', 'Bear')

    cpu_cores = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        # 將所有參數打包送入背景運算
        futures = [executor.submit(_pure_calc_worker, t, df_raw, 
                                   gap_pct=gap_pct, use_sma=use_sma, 
                                   bt_mode=bt_mode, bt_val=bt_val, 
                                   slippage_pct=slippage_pct, fee_disc=fee_disc, 
                                   min_vol=min_vol, sector_map=sector_map,
                                   **kwargs) 
                   for t, df_raw in raw_data_dict.items()]
        
        for future in as_completed(futures):
            res_df = future.result()
            if res_df is not None:
                all_trades.append(res_df)

    if not all_trades:
        print("❌ 錯誤：所有標的運算後均無訊號產生，或因參數錯誤導致背景全部崩潰。")
        return pd.DataFrame(), None, pd.DataFrame()

    raw_portfolio_df = pd.concat(all_trades).reset_index(drop=True)
    raw_portfolio_df['真實進場日'] = pd.to_datetime(raw_portfolio_df['日期'])
    
    actual_taken_trades = []
    active_positions = [] 
    
    cash = initial_capital              
    realized_equity = initial_capital   
    peak_equity = initial_capital       
    is_bankrupt = False      

    # 🌟 審計優化：加入 dropna() 確保時間軸絕對乾淨
    all_dates = pd.concat([pd.to_datetime(raw_portfolio_df['日期']), pd.to_datetime(raw_portfolio_df['出場日期'])]).dropna().unique()
    timeline = sorted(all_dates)
    
    signals_by_date = dict(tuple(raw_portfolio_df.groupby('真實進場日')))

    for current_date in timeline:
        if is_bankrupt: break
        
        still_active = []
        for pos in active_positions:
            if pos['exit_time'] <= current_date:
                cash += pos['returned_cash']
                realized_equity += pos['net_profit']
            else:
                still_active.append(pos)
        active_positions = still_active
        
        peak_equity = max(peak_equity, realized_equity)
        current_dd = (peak_equity - realized_equity) / peak_equity
        
        if current_dd >= account_dd_limit:
            risk_multiplier = 0.2 
        elif current_dd >= account_dd_limit / 2:
            risk_multiplier = 0.5 
        else:
            risk_multiplier = 1.0

        if realized_equity <= 0:
            is_bankrupt = True
            break

        if current_date in signals_by_date and risk_multiplier > 0:
            daily_signals = signals_by_date[current_date].copy()
            daily_signals = daily_signals.sort_values(by='Alpha動能分數', ascending=False)
            
            try:
                m_idx = market_df.index.get_indexer([current_date], method='pad')[0]
                m_status = market_df.iloc[m_idx]
                current_regime = m_status['Regime'] + "_" + ("HighVol" if m_status['Volatility_20d'] > 25.0 else "LowVol")
            except:
                current_regime = "Unknown"

            for _, trade in daily_signals.iterrows():
                trade_sec = trade.get('板塊', 'Unknown') 
                
                if len(active_positions) >= max_concurrent:
                    continue 

                current_sec_count = sum(1 for p in active_positions if p['sector'] == trade_sec)
                if current_sec_count >= max_per_sector:
                    continue

                active_risk_amount = realized_equity * risk_decimal * risk_multiplier
                stop_dist = trade.get('每股風險', trade['進場'] * 0.05)
                if stop_dist <= 0: continue
                
                planned_shares = int(active_risk_amount / stop_dist)
                if planned_shares <= 0: continue
                
                required_cash = planned_shares * trade['進場']
                
                # 流動性過濾
                vol = trade.get('Volume', 0)
                close_p = trade['進場']
                if pd.isna(vol) or vol <= 0:
                    liquidity_limit_cash = 300000 
                else:
                    liquidity_limit_cash = max((vol * close_p) * 0.05, 300000)
                    
                if required_cash > liquidity_limit_cash:
                    planned_shares = int(liquidity_limit_cash / trade['進場'])
                    required_cash = planned_shares * trade['進場']

                if required_cash > cash:
                    planned_shares = int(cash / trade['進場'])
                    required_cash = planned_shares * trade['進場']
                    
                if planned_shares <= 0: continue 

                cash -= required_cash
                
                real_r = trade['真實R']
                actual_risk_taken = planned_shares * stop_dist
                expected_net_profit = actual_risk_taken * real_r
                returned_cash = required_cash + expected_net_profit
                
                # 🌟 審計修復：確實補上 exit_time 的宣告！
                exit_time = pd.to_datetime(trade.get('出場日期'))
                
                active_positions.append({
                    'exit_time': exit_time, 
                    'sector': trade_sec,
                    'invested_cash': required_cash,
                    'returned_cash': returned_cash,
                    'net_profit': expected_net_profit
                })
                
                actual_taken_trades.append({
                    '日期': current_date.date(),
                    '出場日期': exit_time.date(),
                    '標的': trade['標的'],
                    '板塊': trade_sec, 
                    'Regime': current_regime, 
                    '真實R': real_r,
                    '投入資金': round(required_cash, 2),
                    '絕對損益': round(expected_net_profit, 2),
                    '進場時淨值': round(realized_equity, 2)
                })

    if not actual_taken_trades:
        print("❌ 錯誤：所有訊號在進入事件迴圈後，都被資金或風控條件強行擋下。")
        return pd.DataFrame(), None, pd.DataFrame()

    if not is_bankrupt:
        for pos in active_positions:
            cash += pos['returned_cash']
            realized_equity += pos['net_profit']

    portfolio_df = pd.DataFrame(actual_taken_trades)
    portfolio_df = portfolio_df.sort_values(by='出場日期').reset_index(drop=True)
    portfolio_df['交易後總淨值'] = initial_capital + portfolio_df['絕對損益'].cumsum()
    
    final_return_pct = ((realized_equity / initial_capital) - 1) * 100.0
    
    portfolio_df['真實回撤(%)'] = ((portfolio_df['交易後總淨值'] - portfolio_df['交易後總淨值'].expanding().max()) / portfolio_df['交易後總淨值'].expanding().max()) * 100.0
    max_dd_pct = abs(portfolio_df['真實回撤(%)'].min()) if not portfolio_df['真實回撤(%)'].empty else 0.0
    
    bm_df = market_df.copy()
    if bm_df is not None and not bm_df.empty:
        start_date = portfolio_df['日期'].min()
        end_date = portfolio_df['出場日期'].max()
        bm_df = bm_df.loc[start_date:end_date].copy()
        if not bm_df.empty:
            base_price = bm_df['Close'].iloc[0]
            bm_df['基準報酬(%)'] = ((bm_df['Close'] / base_price) - 1) * 100
            bm_df = bm_df.reset_index()

    total_trades = len(portfolio_df)
    
    if total_trades > 0:
        win_rate = len(portfolio_df[portfolio_df['真實R'] > 0]) / total_trades
        avg_w = portfolio_df[portfolio_df['真實R'] > 0]['真實R'].mean() if win_rate > 0 else 0
        avg_l = abs(portfolio_df[portfolio_df['真實R'] < 0]['真實R'].mean()) if win_rate < 1 else 1
        
        # 1. 獲利因子 (Profit Factor) = 總毛利 / 總毛損
        gross_profit = portfolio_df[portfolio_df['絕對損益'] > 0]['絕對損益'].sum()
        gross_loss = abs(portfolio_df[portfolio_df['絕對損益'] < 0]['絕對損益'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 2. 平均 R 倍數 (Average R multiple)
        avg_r = portfolio_df['真實R'].mean()
        
        # 3. 投資組合夏普比率 (Sharpe Ratio)
        # 建立真實的日淨值曲線來精準計算組合夏普
        daily_equity = portfolio_df.groupby('出場日期')['交易後總淨值'].last()
        if not daily_equity.empty and len(daily_equity) > 1:
            idx = pd.date_range(daily_equity.index.min(), daily_equity.index.max())
            daily_equity = daily_equity.reindex(idx, method='ffill')
            daily_returns = daily_equity.pct_change().dropna()
            daily_vol = daily_returns.std()
            sharpe_ratio = (daily_returns.mean() / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0.0
        else:
            sharpe_ratio = 0.0
            
    else:
        win_rate, avg_w, avg_l, avg_r, profit_factor, sharpe_ratio = 0, 0, 0, 0, 0, 0

    
    stats = {
        "總交易次數": total_trades,
        "風控與資金不足淘汰次數": len(raw_portfolio_df) - total_trades if 'raw_portfolio_df' in locals() else 0,
        "勝率(%)": round(win_rate * 100, 2),
        "Portfolio期望值(R)": round((win_rate * avg_w) - ((1 - win_rate) * avg_l), 2),
        "平均R倍數": round(avg_r, 2),
        "獲利因子(PF)": round(profit_factor, 2) if profit_factor != float('inf') else "無限大",
        "夏普值(Sharpe)": round(sharpe_ratio, 2),
        "真實總報酬(%)": round(final_return_pct, 2),
        "期末總淨值(NTD)": round(realized_equity, 0),
        "真實最大回撤(%)": round(max_dd_pct, 2)
    }
    
    if is_bankrupt:
        stats["真實總報酬(%)"] = -100.0
        stats["期末總淨值(NTD)"] = 0
        stats["Portfolio期望值(R)"] = "破產歸零"
        stats["夏普值(Sharpe)"] = "破產"
        stats["獲利因子(PF)"] = 0.0
    
    print("\n🔍 [DEBUG 終端機回報] 算出來的 Stats 字典是:")
    print(stats)
    print("===============================\n")
    
    return portfolio_df, stats, bm_df
