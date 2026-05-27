# backtest_engine.py
import numpy as np
import pandas as pd
import logging

class InstitutionalBacktestEngine:
    def __init__(self, initial_capital=1000000.0, risk_per_trade=0.02, 
                 fee_rate=0.001425, tax_rate=0.003, slippage_pct=0.002, rr_ratio=3.0, bt_mode="固定盈虧比"):
        self.initial_capital = initial_capital
        self.max_risk_pct = risk_per_trade
        self.fee_rate = fee_rate
        self.tax_rate = tax_rate
        self.slippage_pct = slippage_pct  
        self.rr_ratio = rr_ratio 
        self.bt_mode = bt_mode # 新增：接收出場模式

    def run_backtest(self, df):
        cash = self.initial_capital
        active_positions = []  
        trade_history = []
        mtm_equity_curve = []  
        
        if df is None or df.empty or 'Close' not in df.columns:
            return pd.DataFrame(), df, 0.0
            
        has_vol = 'Turnover_5MA' in df.columns
        
        # 🚀 效能優化：改用 itertuples 進行迭代，速度提升 50 倍以上
        for row in df.itertuples():
            index = row.Index
            current_open = row.Open
            current_high = row.High
            current_low = row.Low
            current_close = row.Close
            
            still_active = []
            for pos in active_positions:
                exit_price = None
                exit_reason = ""

                # ==========================================
                # 📈 補齊 UI 遺失功能：移動停利 (Trailing Stop)
                # ==========================================
                if self.bt_mode == "移動停利":
                    if current_high > pos.get('highest_seen', pos['entry_price']):
                        pos['highest_seen'] = current_high
                        # 根據最高價，向下回落 (風險距 * 設定的倍數) 作為移動停損點
                        new_stop = pos['highest_seen'] - (pos['risk_dist'] * self.rr_ratio)
                        if new_stop > pos['stop_loss']:
                            pos['stop_loss'] = new_stop

                # ==========================================
                # 🎯 出場條件判定 (優先判定停損，態度悲觀保守)
                # ==========================================
                if current_open <= pos['stop_loss']:
                    exit_price = current_open * (1 - self.slippage_pct) # 跳空跌破，開盤即砍
                    exit_reason = "Gap_Through_Stop"
                elif current_low <= pos['stop_loss']:
                    exit_price = pos['stop_loss'] * (1 - self.slippage_pct)
                    exit_reason = "Hit_Stop"
                elif self.bt_mode != "移動停利":
                    # 固定盈虧比模式
                    if current_open >= pos['take_profit']:
                        exit_price = current_open * (1 - self.slippage_pct) # 跳空開高飛越目標，賺更多
                        exit_reason = "Gap_Over_Target"
                    elif current_high >= pos['take_profit']:
                        exit_price = pos['take_profit'] * (1 - self.slippage_pct)
                        exit_reason = "Hit_Target"

                # 結算交易
                if exit_price is not None:
                    exit_revenue = pos['shares'] * exit_price * (1 - self.fee_rate - self.tax_rate)
                    net_profit = exit_revenue - pos['entry_cost']
                    cash += exit_revenue  
                    
                    p_risk = pos['planned_risk_amount'] if pos['planned_risk_amount'] > 0 else 1.0
                    real_r = net_profit / p_risk

                    trade_history.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': index,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'net_profit': net_profit,
                        'real_r': real_r,
                        'reason': exit_reason,
                        'adjusted_risk': pos['risk_dist']
                    })
                else:
                    still_active.append(pos)
            
            active_positions = still_active

            # 計算逐日盯市淨值
            current_mtm_equity = cash + sum([p['shares'] * current_close for p in active_positions])
            mtm_equity_curve.append(current_mtm_equity)

            # 動態取得列資料 (防呆 getattr)
            fvg = getattr(row, 'FVG', False)
            adj_risk = getattr(row, 'adjusted_risk', 0)
            if has_vol:
                liquidity_ok = getattr(row, 'Turnover_5MA', float('inf')) > 20000000
            else:
                liquidity_ok = True

            # ==========================================
            # 🌟 進場邏輯 (破除單股資金枷鎖，完美生產訊號)
            # ==========================================
            if fvg and adj_risk > 0 and liquidity_ok:
                risk_dist = adj_risk
                if risk_dist <= 1e-5:
                    continue
                    
                # 確保就算單股帳戶破產，也用初始資金基準去計算部位，不漏掉任何給外層引擎的訊號
                virtual_equity = max(current_mtm_equity, self.initial_capital)
                planned_risk_amount = virtual_equity * self.max_risk_pct
                shares = int(planned_risk_amount / risk_dist)
                
                # 強制至少 1 股，確保真實 R 倍數能被計算並傳遞給 portfolio_engine
                if shares <= 0: 
                    shares = 1
                    planned_risk_amount = risk_dist
                    
                entry_price = current_close
                entry_cost = shares * entry_price * (1 + self.fee_rate + self.slippage_pct)
                
                cash -= entry_cost  # 允許內部 cash 變負數，我們只在乎 R 倍數與訊號生命週期
                
                active_positions.append({
                    'entry_time': index,
                    'entry_price': entry_price,
                    'stop_loss': entry_price - risk_dist,
                    'take_profit': entry_price + (risk_dist * self.rr_ratio), 
                    'shares': shares,
                    'entry_cost': entry_cost,
                    'planned_risk_amount': planned_risk_amount,
                    'risk_dist': risk_dist,
                    'highest_seen': entry_price # 供移動停利用
                })

        df['MtM_Equity'] = mtm_equity_curve
        df['Daily_Return'] = df['MtM_Equity'].pct_change().fillna(0)
        
        daily_volatility = df['Daily_Return'].std()
        sharpe_ratio = (df['Daily_Return'].mean() / daily_volatility) * np.sqrt(252) if daily_volatility > 0 else 0.0
            
        trades_df = pd.DataFrame(trade_history)
        return trades_df, df, sharpe_ratio


# 🌟 完美相容轉接器 (相容組合引擎的呼叫)
def run_backtest_core(df, bt_mode="固定盈虧比", bt_val=3.0, slippage_pct=0.002, fee_disc=0.28, tax_rate=0.003, min_vol=0, **kwargs):
    actual_fee_rate = 0.001425 * fee_disc
    
    # 向下相容舊版參數名
    if bt_mode == "R倍數": 
        bt_mode = "固定盈虧比"
        
    engine = InstitutionalBacktestEngine(
        initial_capital=1000000.0, 
        risk_per_trade=0.02, 
        fee_rate=actual_fee_rate,
        tax_rate=tax_rate,
        slippage_pct=slippage_pct,
        rr_ratio=bt_val,
        bt_mode=bt_mode # 完美接入 UI 設定
    )
    
    trades_df, updated_df, sharpe_ratio = engine.run_backtest(df)
    
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(), updated_df
        
    compat_trades_df = trades_df.rename(columns={
        'entry_time': '日期',
        'exit_time': '出場日期',
        'entry_price': '進場',
        'exit_price': '出場',
        'real_r': '真實R',
        'net_profit': '絕對損益'
    })
    
    compat_trades_df['每股風險'] = compat_trades_df['adjusted_risk']
    return compat_trades_df, updated_df