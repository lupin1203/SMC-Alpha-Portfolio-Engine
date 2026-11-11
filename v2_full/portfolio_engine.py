# portfolio_engine.py
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import pandas as pd
import numpy as np
import logging
import traceback
from config import FEE_RATE, DEFAULT_FEE_DISC
from strategy import apply_smc_logic
from backtest_engine import InstitutionalBacktestEngine, _rename_trades
from data_provider import fetch_stock_data

def _to_lots(shares):
    """強制整張：無條件捨去到 1000 股的倍數（台股一張 = 1000 股）"""
    return (int(shares) // 1000) * 1000

def _pure_calc_worker(ticker, df_raw, **kwargs):
    try:
        df_raw = df_raw.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume']).copy()
        if len(df_raw) < 60: return None

        if df_raw.index.tz is not None:
            df_raw.index = df_raw.index.tz_localize(None)

        df_raw['Vol_MA20'] = df_raw['Volume'].rolling(20).mean()
        df_raw['Score_VolSurge'] = df_raw['Volume'] / df_raw['Vol_MA20']
        
        gap_pct = kwargs.get('gap_pct', 0.3)
        use_sma = kwargs.get('use_sma', True)
        
        df_logic = apply_smc_logic(
            df_raw, 
            min_gap_pct=gap_pct, 
            use_sma_filter=use_sma,
            **kwargs 
        )
        
        if df_logic.index.tz is not None:
            df_logic.index = df_logic.index.tz_localize(None)
            
        years = kwargs.get('years', 10)
        if not kwargs.get('_skip_year_cutoff', False):
            cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=years * 365)
            df_logic = df_logic[df_logic.index >= cutoff_date]
        
        if len(df_logic) < 20: return None

        # ✅ 從 kwargs 取出預先抓好的基本面快取（不在 worker 裡打 API）
        fundamental_cache = kwargs.get('fundamental_cache', {})

        engine = InstitutionalBacktestEngine(
            initial_capital=kwargs.get('initial_capital', 1000000.0),
            risk_per_trade=kwargs.get('risk_pct', 2.0) / 100.0,
            fee_rate=FEE_RATE * kwargs.get('fee_disc', DEFAULT_FEE_DISC),
            slippage_pct=kwargs.get('slippage_pct', 0.002),
            rr_ratio=kwargs.get('bt_val', 3.0),
            bt_mode=kwargs.get('bt_mode', 'R倍數'),
            fundamental_cache=fundamental_cache,
            min_holding_days=kwargs.get('min_holding_days', 0),
            entry_min_yoy=kwargs.get('entry_min_yoy', None),
            entry_require_gm=kwargs.get('entry_require_gm', False),
            entry_require_ocf=kwargs.get('entry_require_ocf', False),
        )
        
        res = engine.run_backtest(df_logic, ticker=ticker)
        
        if res[0] is not None and not res[0].empty:
            trades_df = _rename_trades(res[0].copy())  # ✅ 統一使用共用 rename，不再各自維護
            
            trades_df['標的'] = ticker
            sec_map = kwargs.get('sector_map')
            if sec_map is None:
                sec_map = {}
            trades_df['板塊'] = sec_map.get(ticker, 'Unknown')
            
            entry_dates = pd.to_datetime(trades_df['日期']).dt.tz_localize(None)
            
            if 'Score_VolSurge' in df_raw.columns:
                idx_array = df_raw.index.get_indexer(entry_dates, method='nearest')
                trades_df['Alpha動能分數'] = df_raw['Score_VolSurge'].iloc[idx_array].values
            else:
                trades_df['Alpha動能分數'] = 1.0
                
            trades_df['Volume'] = df_raw['Volume'].iloc[df_raw.index.get_indexer(entry_dates, method='nearest')].values 
            
            if '每股風險' not in trades_df.columns:
                trades_df['每股風險'] = trades_df['進場'] * 0.05
                
            return trades_df
            
    except Exception as e:
        print(f"🔥 單股運算崩潰 ({ticker}): {e}")
        logging.error(traceback.format_exc())
    return None

def run_portfolio_backtest(tickers, years, bt_mode, bt_val, gap_pct, use_sma, slippage_pct, fee_disc, min_vol, max_concurrent=5, max_per_sector=2, sector_map=None, risk_pct=2.0, initial_capital=1_000_000.0, account_dd_limit=0.20, max_position_pct=0.15, max_corr=0.75,
    allowed_regimes=None,
    min_holding_days=0,
    regime_benchmark="0050.TW",   # Regime 判斷基準：0050.TW / TAIEX / TPEx
    vol_threshold=25.0,           # 高波動門檻（年化波動率 %）
    enable_sector_rotation=False, # ✅ 啟用動態板塊輪動（每月選最強 N 板塊）
    sector_top_n=3,               # 選最強幾個板塊
    full_sector_map=None,         # {板塊名: [tickers]}，輪動評分用全板塊地圖
    _preloaded_data=None,         # WFO 預載的 K 線 dict，有的話跳過磁碟讀取
    _preloaded_fundamental=None,  # WFO 預載的基本面 dict
    **kwargs):
    
    print(f"\n==========================================")
    print(f"⚙️ [組合回測引擎] 啟動！接收 {len(tickers)} 檔標的。")
    print(f"==========================================")

    risk_decimal = risk_pct / 100.0
    raw_data_dict = {}
    all_trades = []

    # ✅ WFO 預載模式：資料已切好日期範圍，不再做 years 截斷
    if _preloaded_data is not None:
        for t, df in _preloaded_data.items():
            if t in tickers and len(df) >= 100:
                raw_data_dict[t] = df
        logging.info(f"✅ WFO 預載模式：直接使用 {len(raw_data_dict)} 檔本地 K 線（不截斷）")
    else:
        def _fetch_worker(t):
            df = fetch_stock_data(t, days=years * 365)
            if df is not None and not df.empty:
                df = df.tail(int(years * 252))
                if len(df) >= 100:
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

    # ✅ 相關矩陣
    try:
        returns_df = pd.DataFrame({
            t: df['Close'].pct_change()
            for t, df in raw_data_dict.items()
            if len(df) > 60
        }).dropna(how='all')
        corr_matrix = returns_df.corr()
        logging.info(f"✅ 相關矩陣計算完成 ({corr_matrix.shape[0]}×{corr_matrix.shape[1]})")
    except Exception as e:
        corr_matrix = pd.DataFrame()
        logging.warning(f"相關矩陣計算失敗，略過相關性風控: {e}")

    # ✅ 基本面序列：WFO 預載模式直接用，否則重新抓
    if _preloaded_fundamental is not None:
        fundamental_cache = _preloaded_fundamental
        logging.info(f"✅ WFO 預載模式：直接使用 {len(fundamental_cache)} 檔基本面序列")
    else:
        fundamental_cache = {}
        try:
            from revenue_fetcher import build_fundamental_history_cache
            logging.info(f"📊 預載 {len(raw_data_dict)} 檔歷史基本面序列...")
            fundamental_cache = build_fundamental_history_cache(
                list(raw_data_dict.keys()), delay=0.2
            )
            logging.info("✅ 歷史基本面序列載入完成")
        except Exception as e:
            logging.warning(f"基本面歷史序列預載失敗，略過基本面出場邏輯: {e}")

    # Regime 基準：0050 走本地 yfinance 快取，TAIEX/TPEx 走 FinMind
    if regime_benchmark in ("TAIEX", "TPEx"):
        try:
            from revenue_fetcher import fetch_taiwan_index
            market_df = fetch_taiwan_index(regime_benchmark, days=years*365)
            # FinMind 限流或無資料會回傳空 → 退回 0050
            if market_df is None or market_df.empty:
                logging.warning(f"FinMind 指數 {regime_benchmark} 回傳空資料（可能限流），改用 0050")
                market_df = fetch_stock_data("0050.TW", days=years*365)
        except Exception as e:
            logging.warning(f"FinMind 指數 {regime_benchmark} 抓取失敗，改用 0050：{e}")
            market_df = fetch_stock_data("0050.TW", days=years*365)
    else:
        market_df = fetch_stock_data("0050.TW", days=years*365)
    if market_df is not None and not market_df.empty:
        # 防呆：索引可能是 RangeIndex（FinMind 回來未設日期索引），先轉成 DatetimeIndex
        if not isinstance(market_df.index, pd.DatetimeIndex):
            date_col = None
            for c in ['date', 'Date', 'datetime', 'index']:
                if c in market_df.columns:
                    date_col = c
                    break
            if date_col is not None:
                market_df = market_df.set_index(date_col)
            market_df.index = pd.to_datetime(market_df.index, errors='coerce')
            market_df = market_df[market_df.index.notna()]
        if isinstance(market_df.index, pd.DatetimeIndex) and market_df.index.tz is not None:
            market_df.index = market_df.index.tz_localize(None)
        market_df['SMA60'] = market_df['Close'].rolling(60).mean()
        market_df['SMA200'] = market_df['Close'].rolling(200).mean()
        market_df['Daily_Return'] = market_df['Close'].pct_change()
        market_df['Volatility_20d'] = market_df['Daily_Return'].rolling(20).std() * np.sqrt(252) * 100
        market_df['Regime'] = np.where(market_df['Close'] > market_df['SMA200'], 'Bull', 'Bear')

    # ✅ WFO 預載模式用單進程避免 ProcessPoolExecutor 在重複呼叫中耗盡資源
    #    一般模式用多進程加速
    if _preloaded_data is not None:
        for t, df_raw in raw_data_dict.items():
            res_df = _pure_calc_worker(
                t, df_raw,
                gap_pct=gap_pct, use_sma=use_sma,
                bt_mode=bt_mode, bt_val=bt_val,
                slippage_pct=slippage_pct, fee_disc=fee_disc,
                min_vol=min_vol, sector_map=sector_map,
                years=years, initial_capital=initial_capital,
                risk_pct=risk_pct,
                min_holding_days=min_holding_days,
                fundamental_cache=fundamental_cache,
                _skip_year_cutoff=True,   # WFO 模式不截斷
                **kwargs
            )
            if res_df is not None:
                all_trades.append(res_df)
    else:
        cpu_cores = max(1, multiprocessing.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            futures = [executor.submit(_pure_calc_worker, t, df_raw,
                                       gap_pct=gap_pct, use_sma=use_sma,
                                       bt_mode=bt_mode, bt_val=bt_val,
                                       slippage_pct=slippage_pct, fee_disc=fee_disc,
                                       min_vol=min_vol, sector_map=sector_map,
                                       years=years, initial_capital=initial_capital,
                                       risk_pct=risk_pct,
                                       min_holding_days=min_holding_days,
                                       fundamental_cache=fundamental_cache,
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
    raw_portfolio_df['真實進場日'] = pd.to_datetime(raw_portfolio_df['日期'], errors='coerce')
    raw_portfolio_df = raw_portfolio_df.dropna(subset=['真實進場日'])
    
    actual_taken_trades = []
    _exit_equity_map = {}   # {trade_idx: realized_equity at exit}
    active_positions = [] 
    
    cash = initial_capital              
    realized_equity = initial_capital   
    peak_equity = initial_capital       
    is_bankrupt = False      

    all_dates = pd.concat([
        pd.to_datetime(raw_portfolio_df['日期'], errors='coerce'), 
        pd.to_datetime(raw_portfolio_df['出場日期'], errors='coerce')
    ]).dropna().unique()
    
    timeline = sorted(all_dates)
    signals_by_date = dict(tuple(raw_portfolio_df.groupby('真實進場日')))

    # 動態板塊輪動狀態（用 list 當可變容器，迴圈內每月更新一次）
    _sector_rotation_month = [None]   # 上次計算的 (年, 月)
    _sector_rotation_top   = [None]   # 當下最強板塊名稱集合

    for current_date in timeline:
        if is_bankrupt: break

        still_active = []
        for pos in active_positions:
            if pos['exit_time'] <= current_date:
                cash += pos['returned_cash']
                realized_equity += pos['net_profit']
                if 'trade_idx' in pos:
                    _exit_equity_map[pos['trade_idx']] = realized_equity
                    # ✅ 出場時更新 reason：完全信任個股引擎給的出場理由。
                    #    backtest_engine 已在出場當下正確區分：
                    #      Trailing_Stop（移動停利鎖利）vs Hit_Stop（真停損）
                    #      Gap_Through_Stop / Catastrophic_Stop / Fundamental_Exit / Hit_Target
                    #    不再用 net_r 反推（舊版反推會把賺錢的移動停利錯標成停損，
                    #    污染歸因分析）。
                    idx = pos['trade_idx']
                    if idx < len(actual_taken_trades):
                        net_r = actual_taken_trades[idx]['真實R']
                        signal_reason = pos.get('signal_reason', '') or ''
                        if signal_reason:
                            # 把引擎的 Trailing_Stop / Hit_Target / Gap_Over_Target
                            # 統一顯示為「移動停利」，其餘原樣保留
                            if signal_reason in ("Trailing_Stop", "Hit_Target", "Gap_Over_Target"):
                                actual_taken_trades[idx]['reason'] = "移動停利"
                            else:
                                actual_taken_trades[idx]['reason'] = signal_reason
                        else:
                            # 極少數無標籤的情況才用 R 值補判斷
                            if net_r > 0:
                                actual_taken_trades[idx]['reason'] = "移動停利"
                            else:
                                actual_taken_trades[idx]['reason'] = "Hit_Stop"
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
            
            if 'Alpha動能分數' in daily_signals.columns and daily_signals['Alpha動能分數'].notna().any():
                daily_signals = daily_signals.sort_values(by='Alpha動能分數', ascending=False)
            elif '每股風險' in daily_signals.columns and daily_signals['每股風險'].notna().any():
                daily_signals = daily_signals.sort_values(by='每股風險', ascending=True)
            else:
                safe_seed = int(pd.Timestamp(current_date).strftime('%Y%m%d'))
                daily_signals = daily_signals.sample(frac=1, random_state=safe_seed)
            
            try:
                m_idx = market_df.index.get_indexer([current_date], method='pad')[0]
                m_status = market_df.iloc[m_idx]
                current_regime = m_status['Regime'] + "_" + ("HighVol" if m_status['Volatility_20d'] > vol_threshold else "LowVol")
            except:
                current_regime = "Unknown"

            for _, trade in daily_signals.iterrows():
                trade_sec = trade.get('板塊', 'Unknown')

                # ✅ Regime 過濾：只在允許的市場環境進場
                if allowed_regimes and current_regime not in allowed_regimes:
                    continue

                # ✅ 動態板塊輪動：只在「當下最強 N 板塊」進場（point-in-time，無前視偏差）
                if enable_sector_rotation and full_sector_map and _preloaded_fundamental:
                    cur_month = (pd.Timestamp(current_date).year, pd.Timestamp(current_date).month)
                    if cur_month != _sector_rotation_month[0]:
                        # 每月重算一次最強板塊（用當下已公告的月營收）
                        try:
                            from revenue_fetcher import rank_sectors_at_date
                            _sector_rotation_top[0] = set(rank_sectors_at_date(
                                full_sector_map, current_date, _preloaded_fundamental,
                                top_n=sector_top_n))
                            _sector_rotation_month[0] = cur_month
                        except Exception:
                            _sector_rotation_top[0] = None  # 算失敗就不過濾（保守）
                    if _sector_rotation_top[0] and trade_sec not in _sector_rotation_top[0]:
                        continue  # 這個板塊不在當下最強名單，跳過
                
                if len(active_positions) >= max_concurrent:
                    continue

                current_sec_count = sum(1 for p in active_positions if p['sector'] == trade_sec)
                if current_sec_count >= max_per_sector:
                    continue

                # ✅ 相關性風控：新進標的與現有持倉的相關係數不得超過 max_corr
                if not corr_matrix.empty and active_positions:
                    high_corr = any(
                        abs(corr_matrix.loc[trade['標的'], pos.get('ticker', '')]) > max_corr
                        for pos in active_positions
                        if pos.get('ticker', '') and trade['標的'] in corr_matrix.columns
                        and pos.get('ticker', '') in corr_matrix.columns
                    )
                    if high_corr:
                        continue

                active_risk_amount = realized_equity * risk_decimal * risk_multiplier
                stop_dist = trade.get('每股風險', trade['進場'] * 0.05)
                if stop_dist <= 0: continue

                planned_shares = _to_lots(active_risk_amount / stop_dist)
                if planned_shares <= 0: continue

                required_cash = planned_shares * trade['進場']

                vol = trade.get('Volume', 0)
                close_p = trade['進場']
                if pd.isna(vol) or vol <= 0:
                    liquidity_limit_cash = 300000
                else:
                    liquidity_limit_cash = max((vol * close_p) * 0.05, 300000)

                if required_cash > liquidity_limit_cash:
                    planned_shares = _to_lots(liquidity_limit_cash / trade['進場'])
                    required_cash = planned_shares * trade['進場']

                if required_cash > cash:
                    planned_shares = _to_lots(cash / trade['進場'])
                    required_cash = planned_shares * trade['進場']

                # ✅ 單一持倉上限：不超過帳戶淨值的 max_position_pct
                max_single_position = realized_equity * max_position_pct
                if required_cash > max_single_position:
                    planned_shares = _to_lots(max_single_position / trade['進場'])
                    required_cash  = planned_shares * trade['進場']

                if planned_shares <= 0: continue

                cash -= required_cash
                
                real_r = trade['真實R']
                actual_risk_taken = planned_shares * stop_dist
                expected_net_profit = actual_risk_taken * real_r
                returned_cash = required_cash + expected_net_profit
                
                exit_time = pd.to_datetime(trade.get('出場日期'))
                
                _trade_idx = len(actual_taken_trades)

                active_positions.append({
                    'exit_time': exit_time,
                    'sector': trade_sec,
                    'ticker': trade['標的'],
                    'invested_cash': required_cash,
                    'returned_cash': returned_cash,
                    'net_profit': expected_net_profit,
                    'trade_idx': _trade_idx,
                    'signal_reason': trade.get('reason', ''),  # 個股引擎的出場原因（參考用）
                })

                actual_taken_trades.append({
                    '日期': current_date.date(),
                    '出場日期': exit_time.date(),
                    '標的': trade['標的'],
                    '板塊': trade_sec,
                    'Regime': current_regime,
                    '進場價': trade.get('進場', 0.0),
                    '防彈止損價': trade.get('進場', 0.0) - trade.get('每股風險', 0.0),
                    'Alpha動能分數': trade.get('Alpha動能分數', 0.0),
                    '真實R': real_r,
                    '投入資金': round(required_cash, 2),
                    '絕對損益': round(expected_net_profit, 2),
                    '進場時淨值': round(realized_equity, 2),
                    'reason': trade.get('reason', ''),  # 暫存個股引擎的 reason，出場時更新
                })

    if not actual_taken_trades:
        print("❌ 錯誤：所有訊號在進入事件迴圈後，都被資金或風控條件強行擋下。")
        return pd.DataFrame(), None, pd.DataFrame()

    if not is_bankrupt:
        for pos in active_positions:
            cash += pos['returned_cash']
            realized_equity += pos['net_profit']

    portfolio_df = pd.DataFrame(actual_taken_trades)
    # ✅ 排序前先把 _exit_equity_map 的值附加到 df，避免 sort 後 index 錯位
    portfolio_df['_exit_eq'] = portfolio_df.index.map(
        lambda i: _exit_equity_map.get(i, None)
    )
    portfolio_df = portfolio_df.sort_values(by='出場日期').reset_index(drop=True)
    portfolio_df['交易後總淨值'] = portfolio_df['_exit_eq'].fillna(method='ffill').round(0)
    portfolio_df.drop(columns=['_exit_eq'], inplace=True)

    # final_return_pct 與 equity curve 現在同源，不再有兩套數字
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
        
        gross_profit = portfolio_df[portfolio_df['絕對損益'] > 0]['絕對損益'].sum()
        gross_loss = abs(portfolio_df[portfolio_df['絕對損益'] < 0]['絕對損益'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_r = portfolio_df['真實R'].mean()
        
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
    
    if not portfolio_df.empty:
        if '出場日期' in portfolio_df.columns:
            portfolio_df = portfolio_df.sort_values('出場日期').reset_index(drop=True)
            
        cols_to_round_2 = ['進場價', '出場價', '防彈止損價', '真實R']
        for c in cols_to_round_2:
            if c in portfolio_df.columns:
                portfolio_df[c] = portfolio_df[c].astype(float).round(2)
                
        cols_to_round_0 = ['絕對損益', '交易後總淨值', '投入資金']
        for c in cols_to_round_0:
            if c in portfolio_df.columns:
                portfolio_df[c] = portfolio_df[c].astype(float).round(0)

    return portfolio_df, stats, bm_df

# ==========================================
# 組合層級 Walk-Forward Optimization
# ==========================================

def run_portfolio_wfo(
    tickers: list,
    bt_mode: str = "移動停利",
    gap_options: list = None,
    bt_val_options: list = None,
    train_years: int = 3,
    test_years: int = 1,
    slippage_pct: float = 0.002,
    fee_disc: float = 0.28,
    min_vol: int = 1000,
    max_concurrent: int = 5,
    max_per_sector: int = 2,
    sector_map: dict = None,
    risk_pct: float = 2.0,
    initial_capital: float = 1_000_000.0,
    account_dd_limit: float = 0.20,
    max_position_pct: float = 0.15,
    max_corr: float = 0.75,
    entry_min_yoy: float = None,
    entry_require_gm: bool = False,
    entry_require_ocf: bool = False,
    allowed_regimes: list = None,
    enable_sector_rotation: bool = False,
    sector_top_n: int = 3,
    full_sector_map: dict = None,
    progress_cb=None,
    enable_purge: bool = True,      # ★ 資料洩漏防護開關：True=乾淨(選項三), False=舊行為(有洩漏,供對比)
    embargo_days: int = 5,          # ★ 禁運期天數：出場日要早於train_end這麼多天(安全緩衝)
    **kwargs,   # 安全吸收不同版本 app.py 可能多傳的參數（如 entry_max_debt_ratio），避免報錯
) -> tuple:
    """
    組合層級滾動前進分析（Portfolio Walk-Forward Optimization）

    正確邏輯：
      從最早的起始點往最近推進，確保每個盲測期都是當時沒見過的未來資料。
      例如 train_years=3, test_years=1：
        視窗1：訓練 2015-2018 → 盲測 2018-2019
        視窗2：訓練 2016-2019 → 盲測 2019-2020
        ...
        視窗N：訓練 2022-2025 → 盲測 2025-2026

    回傳：(wfo_logs_df, final_stats)
    """
    # 一致性警告：若呼叫端開了負債比率過濾，但本版 WFO 未套用，明確提示避免靜默不一致
    if kwargs.get('entry_max_debt_ratio') is not None:
        logging.warning("[WFO] 收到 entry_max_debt_ratio 但本版 WFO 未套用負債比率過濾，"
                        "WFO 結果將與單次回測不一致。若需要，請告知以補上該過濾。")
    import datetime, os

    if gap_options is None:
        gap_options = [0.2, 0.3, 0.4]
    if bt_val_options is None:
        # 移動停利：擴大到 8~20，涵蓋手動測試發現的甜蜜點區間（15 附近）
        bt_val_options = [8.0, 10.0, 12.0, 15.0, 18.0, 20.0] if bt_mode == "移動停利" else [2.0, 3.0, 4.0]

    # ── 工具函數 ──
    def _calc_exp(df):
        if df is None or df.empty or '真實R' not in df.columns or len(df) == 0:
            return -999.0
        t = len(df)
        w = len(df[df['真實R'] > 0]) / t
        avg_w = df[df['真實R'] > 0]['真實R'].mean() if w > 0 else 0.0
        avg_l = abs(df[df['真實R'] < 0]['真實R'].mean()) if w < 1 else 1.0
        return round((w * avg_w) - ((1 - w) * avg_l), 3)

    def _max_dd_r(df):
        """R 單位的最大回撤"""
        if df is None or df.empty or '真實R' not in df.columns:
            return 0.0
        eq = df['真實R'].cumsum()
        return round((eq - eq.expanding().max()).min(), 3)

    def _normalize_equity(df):
        """
        把一段交易的『交易後總淨值』重新歸零到該期起點。
        修正：盲測期篩掉warmup後，淨值帶著warmup盈虧，基準點會偏移。
        歸零後才能正確算該期的夏普與回撤。回傳新的 df（不改原本）。
        """
        if df is None or df.empty or '交易後總淨值' not in df.columns:
            return df
        d = df.sort_values('出場日期').copy() if '出場日期' in df.columns else df.copy()
        # 用逐筆絕對損益重建「從1_000_000起算」的淨值，消除warmup累積
        if '絕對損益' in d.columns:
            base = 1_000_000.0
            d['交易後總淨值'] = base + d['絕對損益'].cumsum()
        return d

    def _calc_sharpe(df):
        """
        ★ 與單次回測完全同源的夏普：用「每日淨值曲線」算年化夏普(×√252)。
        直接複製 run_portfolio_backtest 第500-506行的算法，確保 WFO 與單次回測用同一把尺。
        需要 df 有 '出場日期' 和 '交易後總淨值' 兩欄。
        """
        df = _normalize_equity(df)
        if df is None or df.empty or '交易後總淨值' not in df.columns or '出場日期' not in df.columns:
            return -999.0
        daily_equity = df.groupby('出場日期')['交易後總淨值'].last()
        if daily_equity.empty or len(daily_equity) <= 1:
            return -999.0
        idx = pd.date_range(pd.to_datetime(daily_equity.index).min(),
                            pd.to_datetime(daily_equity.index).max())
        daily_equity.index = pd.to_datetime(daily_equity.index)
        daily_equity = daily_equity.reindex(idx, method='ffill')
        daily_returns = daily_equity.pct_change().dropna()
        daily_vol = daily_returns.std()
        if daily_vol == 0 or len(daily_returns) < 2:
            return -999.0
        return round((daily_returns.mean() / daily_vol) * np.sqrt(252), 4)

    def _calc_dd_pct(df):
        """
        ★ 與單次回測完全同源的回撤：用「淨值曲線」算最大回撤%。
        直接複製 run_portfolio_backtest 第474-475行的算法。
        """
        df = _normalize_equity(df)
        if df is None or df.empty or '交易後總淨值' not in df.columns:
            return 0.0
        eq = df.sort_values('出場日期')['交易後總淨值'] if '出場日期' in df.columns else df['交易後總淨值']
        peak = eq.expanding().max()
        dd = ((eq - peak) / peak) * 100.0
        return round(abs(dd.min()), 2) if not dd.empty else 0.0

    # ── Step 1：預載所有本地資料（只做一次，多執行緒加速）──
    logging.info("WFO 預載資料中...")
    _preloaded_data = {}
    DATA_DIR = "./local_data"

    # 先掃描有哪些檔案存在（快），再批量讀取
    existing = {}
    for t in tickers:
        stock_id = str(t).replace(".TW.TW", "").replace(".TW", "")
        path = os.path.join(DATA_DIR, f"{stock_id}.TW.parquet")
        if os.path.exists(path):
            existing[t] = path

    if progress_cb:
        progress_cb(0, 1, f"Step 1/2：預載 {len(existing)}/{len(tickers)} 檔 K 線（多執行緒）...")

    def _read_parquet(item):
        t, path = item
        try:
            df = pd.read_parquet(path)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [c.capitalize() if c.lower() != 'fvg' else 'FVG' for c in df.columns]
            return t, df
        except Exception:
            return t, None

    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _asc
    with _TPE(max_workers=16) as ex:
        futs = {ex.submit(_read_parquet, item): item[0] for item in existing.items()}
        done = 0
        for fut in _asc(futs):
            t, df = fut.result()
            if df is not None:
                _preloaded_data[t] = df
            done += 1
            if progress_cb and done % 200 == 0:
                progress_cb(0, 1, f"Step 1/2：K 線預載中 {done}/{len(existing)}...")

    if progress_cb:
        progress_cb(0, 1, f"Step 1/2：K 線完成，載入基本面序列...")

    _preloaded_fundamental = {}
    try:
        from revenue_fetcher import REVENUE_HISTORY_DIR, MARGIN_HISTORY_DIR

        def _read_fundamental(t):
            ticker_pure = str(t).replace(".TW", "")
            rev_path    = os.path.join(REVENUE_HISTORY_DIR, f"{ticker_pure}.parquet")
            margin_path = os.path.join(MARGIN_HISTORY_DIR,  f"{ticker_pure}.parquet")
            rev    = pd.read_parquet(rev_path)    if os.path.exists(rev_path)    else pd.DataFrame()
            margin = pd.read_parquet(margin_path) if os.path.exists(margin_path) else pd.DataFrame()
            return t, {"rev": rev, "margin": margin}

        with _TPE(max_workers=16) as ex:
            futs = {ex.submit(_read_fundamental, t): t for t in tickers}
            for fut in _asc(futs):
                t, fund = fut.result()
                _preloaded_fundamental[t] = fund
    except Exception as e:
        logging.warning(f"基本面預載失敗: {e}")

    logging.info(f"✅ 預載完成：{len(_preloaded_data)} 檔 K 線，{len(_preloaded_fundamental)} 檔基本面")

    # ── Step 2：建立正確的滾動視窗序列（從最早往最近推）──
    today      = datetime.date.today()
    data_start = datetime.date(2015, 1, 1)  # 最早可用資料起點

    # 第一個訓練期結束點 = data_start + train_years
    first_test_start = datetime.date(data_start.year + train_years, data_start.month, data_start.day)
    first_test_end   = datetime.date(first_test_start.year + test_years, first_test_start.month, first_test_start.day)

    if first_test_end > today:
        logging.warning("資料範圍不足以建立任何 WFO 視窗")
        return pd.DataFrame(), None

    # 建立所有視窗
    windows = []
    test_start = first_test_start
    while True:
        test_end   = datetime.date(test_start.year + test_years, test_start.month, test_start.day)
        train_start = datetime.date(test_start.year - train_years, test_start.month, test_start.day)

        if test_end > today:
            break

        windows.append({
            "train_start": train_start,
            "train_end":   test_start,
            "test_start":  test_start,
            "test_end":    test_end,
        })
        # 步進一個盲測期
        test_start = test_end

    if not windows:
        return pd.DataFrame(), None

    total_windows = len(windows)
    logging.info(f"WFO 建立 {total_windows} 個滾動視窗")

    def _run_slice_by_date(train_start_dt, test_end_dt, gap, btval):
        """
        用完整組合回測（含 Regime 過濾、保護期、相關性風控、部位管理）
        跑指定日期範圍。資料從預載切片，繞過 years 截斷。
        """
        start_ts = pd.Timestamp(train_start_dt)
        end_ts   = pd.Timestamp(test_end_dt)

        # 從預載資料切出日期範圍
        sliced_data = {}
        for t, df in _preloaded_data.items():
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            df_slice = df[mask]
            if len(df_slice) >= 100:
                sliced_data[t] = df_slice

        if not sliced_data:
            return pd.DataFrame()

        # 走完整組合回測，保留所有風控邏輯
        pf_df, _, _ = run_portfolio_backtest(
            tickers=list(sliced_data.keys()),
            years=99,  # 不截斷（_skip 由 _preloaded_data 觸發）
            bt_mode=bt_mode,
            bt_val=btval,
            gap_pct=gap,
            use_sma=True,
            slippage_pct=slippage_pct,
            fee_disc=fee_disc,
            min_vol=min_vol,
            max_concurrent=max_concurrent,
            max_per_sector=max_per_sector,
            sector_map=sector_map,
            risk_pct=risk_pct,
            initial_capital=initial_capital,
            account_dd_limit=account_dd_limit,
            max_position_pct=max_position_pct,
            max_corr=max_corr,
            allowed_regimes=allowed_regimes if allowed_regimes is not None else ["Bull_LowVol", "Bear_HighVol"],  # 套用正期望值環境過濾
            min_holding_days=10,                               # 套用保護期
            entry_min_yoy=entry_min_yoy,                        # ✅ 套用進場月營收過濾（與單次回測一致）
            entry_require_gm=entry_require_gm,                  # ✅ 套用進場毛利率過濾
            entry_require_ocf=entry_require_ocf,                # ✅ 套用進場現金流過濾
            enable_sector_rotation=enable_sector_rotation,     # ✅ 套用動態板塊輪動
            sector_top_n=sector_top_n,
            full_sector_map=full_sector_map,
            _preloaded_data=sliced_data,
            _preloaded_fundamental=_preloaded_fundamental,
        )
        return pf_df if pf_df is not None else pd.DataFrame()

    # ── Step 3：跑每個視窗 ──
    if progress_cb:
        progress_cb(0, total_windows, "Step 2/2：開始滾動視窗回測...")

    wfo_logs = []
    all_oos_exp = []   # 期望值標準的盲測交易
    all_oos_shp = []   # 夏普標準的盲測交易

    for w_idx, win in enumerate(windows):
        train_start = win["train_start"]
        train_end   = win["train_end"]
        test_start  = win["test_start"]
        test_end    = win["test_end"]

        if progress_cb:
            progress_cb(w_idx, total_windows,
                        f"視窗 {w_idx+1}/{total_windows}："
                        f"訓練 {train_start}～{train_end} | 盲測 {test_start}～{test_end}")

        logging.info(f"WFO 視窗 {w_idx+1}/{total_windows}: 訓練 {train_start}~{train_end}, 盲測 {test_start}~{test_end}")

        # ── 訓練期網格搜尋（做法A：一次回測，同時記錄期望值與夏普，兩種標準各挑最佳）──
        # 用期望值挑的最佳
        best_exp    = -999.0
        best_gap_e  = gap_options[0]
        best_btval_e = bt_val_options[0]
        # 用夏普挑的最佳
        best_shp    = -999.0
        best_gap_s  = gap_options[0]
        best_btval_s = bt_val_options[0]

        total_combos = len(gap_options) * len(bt_val_options)
        combo_idx    = 0

        for gap in gap_options:
            for btval in bt_val_options:
                combo_idx += 1
                if progress_cb:
                    progress_cb(w_idx, total_windows,
                                f"視窗 {w_idx+1}：訓練期 {combo_idx}/{total_combos}（gap={gap}, bt={btval}）")

                train_df = _run_slice_by_date(train_start, train_end, gap, btval)
                if train_df is None or train_df.empty:
                    continue

                # ── 訓練期交易篩選（含資料洩漏防護 Purging）──
                train_df['_exit_dt']  = pd.to_datetime(train_df['出場日期'], errors='coerce')
                train_df['_entry_dt'] = pd.to_datetime(train_df['日期'], errors='coerce')

                if enable_purge:
                    # ★ 選項三 + embargo：一筆交易能進訓練期,必須「進場和出場都在訓練期內」,
                    #   且出場日要早於 train_end 至少 embargo_days 天(留安全緩衝,避免持有期摸到盲測期)。
                    #   這擋掉了「進場早、出場貼近train_end」的交易把接近盲測期的市場資訊帶進訓練期。
                    _purge_cutoff = pd.Timestamp(train_end) - pd.Timedelta(days=embargo_days)
                    train_slice = train_df[
                        (train_df['_entry_dt'] >= pd.Timestamp(train_start)) &
                        (train_df['_exit_dt']  <  _purge_cutoff)
                    ]
                else:
                    # 舊行為(有洩漏,供對比):只用出場日篩,允許進場早、出場貼邊的交易進訓練期
                    train_slice = train_df[
                        (train_df['_exit_dt'] >= pd.Timestamp(train_start)) &
                        (train_df['_exit_dt'] <  pd.Timestamp(train_end))
                    ]
                if len(train_slice) < 10:
                    continue

                # 同一次回測結果，算兩個指標（共用回測，不重跑）
                exp = _calc_exp(train_slice)
                shp = _calc_sharpe(train_slice)
                if exp > best_exp:
                    best_exp, best_gap_e, best_btval_e = exp, gap, btval
                if shp > best_shp:
                    best_shp, best_gap_s, best_btval_s = shp, gap, btval

        logging.info(f"  [期望值標準] 最佳: gap={best_gap_e}, bt={best_btval_e}, exp={best_exp:.3f}R")
        logging.info(f"  [夏普標準]   最佳: gap={best_gap_s}, bt={best_btval_s}, sharpe={best_shp:.3f}")

        # ── 盲測期：兩種看法各跑一次（期望值選的參數 vs 夏普選的參數）──
        warmup_start = datetime.date(test_start.year, test_start.month, test_start.day) - datetime.timedelta(days=90)

        def _run_oos(gap, btval):
            """用指定參數跑盲測，回傳篩過日期的盲測交易。"""
            oos_full = _run_slice_by_date(warmup_start, test_end, gap, btval)
            if oos_full is None or oos_full.empty:
                return pd.DataFrame()
            oos_full['_exit_dt'] = pd.to_datetime(oos_full['出場日期'], errors='coerce')
            return oos_full[
                (oos_full['_exit_dt'] >= pd.Timestamp(test_start)) &
                (oos_full['_exit_dt'] <  pd.Timestamp(test_end))
            ].copy()

        # 看法一：期望值標準選出的參數
        if progress_cb:
            progress_cb(w_idx, total_windows, f"視窗 {w_idx+1}：盲測[期望值標準] gap={best_gap_e}, bt={best_btval_e}")
        oos_e = _run_oos(best_gap_e, best_btval_e)

        # 看法二：夏普標準選出的參數
        # 若兩種標準選到相同參數，直接重用，不重跑盲測（省一半時間）
        if best_gap_s == best_gap_e and best_btval_s == best_btval_e:
            oos_s = oos_e.copy()
            if progress_cb:
                progress_cb(w_idx, total_windows, f"視窗 {w_idx+1}：兩標準參數相同，重用結果")
        else:
            if progress_cb:
                progress_cb(w_idx, total_windows, f"視窗 {w_idx+1}：盲測[夏普標準] gap={best_gap_s}, bt={best_btval_s}")
            oos_s = _run_oos(best_gap_s, best_btval_s)

        def _summ(oos_df):
            c = len(oos_df)
            return {
                "count": c,
                "exp":  _calc_exp(oos_df) if c > 0 else 0.0,
                "netr": round(oos_df['真實R'].sum(), 2) if c > 0 else 0.0,
                "mdd":  _calc_dd_pct(oos_df) if c > 0 else 0.0,   # 淨值回撤%,與單次回測一致
                "shp":  _calc_sharpe(oos_df) if c > 1 else 0.0,   # 日淨值年化夏普,與單次回測一致
            }
        se = _summ(oos_e)
        ss = _summ(oos_s)

        # 兩種看法的盲測交易分別累積（供最後綜合統計）
        if se["count"] > 0:
            all_oos_exp.append(oos_e)
        if ss["count"] > 0:
            all_oos_shp.append(oos_s)

        wfo_logs.append({
            "盲測期間":          f"{test_start} ~ {test_end}",
            # 期望值標準
            "E_最佳停利":        best_btval_e,
            "E_最佳缺口":        best_gap_e,
            "E_OOS期望值":       round(se["exp"], 2),
            "E_OOS淨利R":        se["netr"],
            "E_OOS回撤R":        se["mdd"],
            # 夏普標準
            "S_最佳停利":        best_btval_s,
            "S_最佳缺口":        best_gap_s,
            "S_OOS期望值":       round(ss["exp"], 2),
            "S_OOS淨利R":        ss["netr"],
            "S_OOS回撤R":        ss["mdd"],
            # ── 向後相容：舊 UI 依賴的欄位（對應期望值標準，維持原本語意）──
            "Train最佳缺口":     best_gap_e,
            "Train最佳停利":     best_btval_e,
            "OOS期望值(R)":      round(se["exp"], 2),
            "OOS淨利(R)":        se["netr"],
            "OOS最大回撤(R)":    se["mdd"],
        })

        if progress_cb:
            progress_cb(w_idx + 1, total_windows,
                        f"視窗 {w_idx+1} 完成：E選bt={best_btval_e}(OOS {se['exp']:.2f}R) | S選bt={best_btval_s}(OOS {ss['exp']:.2f}R)")

    if not wfo_logs:
        return pd.DataFrame(), None

    wfo_logs_df = pd.DataFrame(wfo_logs)
    total_w = len(wfo_logs)

    if not all_oos_exp and not all_oos_shp:
        return wfo_logs_df, None

    def _final(all_oos, logs, exp_key, btval_key):
        """算一種看法的綜合統計 + 每視窗選的停利值分布（判斷穩定性）。"""
        if not all_oos:
            return None
        combined = pd.concat(all_oos, ignore_index=True)
        pos_win  = sum(1 for r in logs if r[exp_key] > 0)
        # 每視窗選的最佳停利值 → 看穩不穩定（一直跳=不可信；集中=可信）
        chosen   = [r[btval_key] for r in logs]
        from collections import Counter
        dist     = dict(Counter(chosen))
        return {
            "總盲測交易次數":    len(combined),
            "總盲測淨利(R)":     round(combined['真實R'].sum(), 2),
            "WFO綜合期望值(R)":  round(_calc_exp(combined), 2),
            "WFO綜合夏普(年化)": round(_calc_sharpe(combined), 3),   # 與單次回測同源
            "WFO最大回撤(%)":    round(_calc_dd_pct(combined), 2),   # 淨值回撤%,與單次回測同源
            "正期望值視窗比例":   f"{pos_win}/{total_w}",
            "各視窗選的停利值分布": dist,   # ★ 穩定性關鍵：集中在某值=可信
        }

    stats_exp = _final(all_oos_exp, wfo_logs, "E_OOS期望值", "E_最佳停利")
    stats_shp = _final(all_oos_shp, wfo_logs, "S_OOS期望值", "S_最佳停利")

    final_stats = {
        "【期望值標準】": stats_exp,
        "【夏普標準】":   stats_shp,
        "說明": "比較兩種看法：綜合期望值/夏普/回撤，以及『各視窗選的停利值分布』"
                "——若分布集中在某值(如15)，代表該值穩定可信；若跳來跳去，代表無穩定最佳值。",
    }
    # ── 向後相容：舊 UI 依賴的頂層 key（對應期望值標準）──
    if stats_exp is not None:
        pos_e = sum(1 for r in wfo_logs if r["OOS期望值(R)"] > 0)
        final_stats["WFO綜合期望值(R)"] = stats_exp["WFO綜合期望值(R)"]
        final_stats["WFO最大回撤(R)"]   = stats_exp["WFO最大回撤(%)"]   # UI舊key沿用,值改為淨值回撤%
        final_stats["總盲測交易次數"]    = stats_exp["總盲測交易次數"]
        final_stats["總盲測淨利(R)"]     = stats_exp["總盲測淨利(R)"]
        final_stats["正期望值視窗比例"]  = stats_exp["正期望值視窗比例"]
        final_stats["策略穩定性評分"]    = round(pos_e / total_w * 100, 1) if total_w > 0 else 0.0

    return wfo_logs_df, final_stats