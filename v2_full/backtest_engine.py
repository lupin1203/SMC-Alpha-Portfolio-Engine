# backtest_engine.py
import numpy as np
import pandas as pd
import logging
from config import FEE_RATE, TAX_RATE, DEFAULT_FEE_DISC

class InstitutionalBacktestEngine:
    def __init__(self, initial_capital=1000000.0, risk_per_trade=0.02,
                 fee_rate=FEE_RATE, tax_rate=TAX_RATE, slippage_pct=0.002, rr_ratio=3.0, bt_mode="固定盈虧比",
                 fundamental_cache: dict = None,
                 min_holding_days: int = 0,
                 min_vol: int = 0,             # 最低日均成交量（張），0 = 不過濾
                 entry_min_yoy: float = None,  # 進場要求月營收 YoY ≥ 此值，None = 不過濾
                 entry_require_gm: bool = False,   # 進場要求毛利率擴張
                 entry_require_ocf: bool = False,
                 entry_max_debt_ratio: float = None): # 進場要求負債比率 ≤ 此值，None = 不過濾
                                                      # 建議值 60%，自動擋掉金融/建設等高槓桿板塊
                                                   # ✅ 已實作：現金流歷史序列由 fundamental_cache 的
                                                   #    "cashflow" 提供，套用季報公告延遲（available_date），
                                                   #    抓不到資料不誤殺，與實盤掃描端邏輯一致。
        self.initial_capital    = initial_capital
        self.max_risk_pct       = risk_per_trade
        self.fee_rate           = fee_rate
        self.tax_rate           = tax_rate
        self.slippage_pct       = slippage_pct
        self.rr_ratio           = rr_ratio
        self.bt_mode            = bt_mode
        self.fundamental_cache  = fundamental_cache or {}
        self.min_holding_days   = min_holding_days
        self.min_vol            = min_vol
        self.entry_min_yoy      = entry_min_yoy
        self.entry_require_gm   = entry_require_gm
        self.entry_require_ocf  = entry_require_ocf
        self.entry_max_debt_ratio = entry_max_debt_ratio

    @staticmethod
    def _dynamic_slippage(price: float, base_slippage: float) -> float:
        """
        依股價動態計算滑價：
        股價越低流動性越差，滑價越高。
        > 500 元：base × 0.5   （大型權值股，流動性極好）
        100-500：base × 1.0   （中型股，標準）
        50-100：  base × 1.5
        < 50 元：  base × 2.5  （小型低價股，流動性差）
        """
        if price > 500:
            return base_slippage * 0.5
        elif price > 100:
            return base_slippage * 1.0
        elif price > 50:
            return base_slippage * 1.5
        else:
            return base_slippage * 2.5

    def run_backtest(self, df, ticker: str = ""):
        cash = self.initial_capital
        active_positions = []  
        trade_history = []
        mtm_equity_curve = []  
        _last_fundamental_check_month = None  # 記錄上次基本面檢查的月份，每月只查一次
        
        # 🛡️ 實戰重構 1：待命訂單 (Pending Orders) 陣列，消除收盤價進場的未來函數
        pending_orders = []
        
        if df is None or df.empty or 'Close' not in df.columns:
            return pd.DataFrame(), df, 0.0
            
        df = df.copy() # 確保修改欄位時不會報錯
        
        # ==========================================
        # 🚀 V20.0 升級：預先計算「一字跌停鎖死」遮罩
        # ==========================================
        df['Prev_Close'] = df['Close'].shift(1)
        # 條件：今天最高價等於最低價 (一字線) 且 收盤價比昨天低 (跌停鎖死)
        df['Is_Limit_Down_Locked'] = (df['High'] == df['Low']) & (df['Close'] < df['Prev_Close'])

        has_vol = 'Turnover_5MA' in df.columns
        
        # 使用 itertuples 進行高速迭代
        for row in df.itertuples():
            index = row.Index
            current_open = row.Open
            current_high = row.High
            current_low = row.Low
            current_close = row.Close
            
            # 👇 讀取今天的「跌停鎖死」標記
            is_locked = getattr(row, 'Is_Limit_Down_Locked', False)
            
            # ==========================================
            # 🎯 階段一：執行昨日收盤確認的「待命進場訂單」 (以今日開盤價 T+1 Open 執行)
            # ==========================================
            new_positions = []
            for order in pending_orders:
                virtual_equity = max(cash + sum([p['shares'] * current_open for p in active_positions]), self.initial_capital)
                planned_risk_amount = virtual_equity * self.max_risk_pct
                
                # 🛡️ 實戰重構 2：台股整張 (1000股) 流動性限制
                # 剔除畸形的 142 股零股交易，強制向下取整至 1000 的倍數
                raw_shares = int(planned_risk_amount / order['risk_dist'])
                shares = (raw_shares // 1000) * 1000 
                
                if shares > 0:
                    # 動態滑價：依進場股價計算
                    entry_slip = self._dynamic_slippage(current_open, self.slippage_pct)
                    slipped_entry_price = current_open * (1 + entry_slip)
                    gross_entry = shares * slipped_entry_price
                    
                    # 🛡️ 實戰重構 3：進場的 20 元台幣最低手續費地板
                    entry_fee = max(20.0, gross_entry * self.fee_rate)
                    entry_cost = gross_entry + entry_fee
                    
                    cash -= entry_cost  
                    
                    new_positions.append({
                        'entry_time': index,
                        'entry_date': index,           # ✅ 記錄進場日期供最短持倉判斷
                        'entry_price': slipped_entry_price,
                        'sl_price_initial': slipped_entry_price - order['risk_dist'],
                        'stop_loss': slipped_entry_price - order['risk_dist'],
                        'take_profit': slipped_entry_price + (order['risk_dist'] * self.rr_ratio),
                        'shares': shares,
                        'entry_cost': entry_cost,
                        'planned_risk_amount': planned_risk_amount,
                        'risk_dist': order['risk_dist'],
                        'highest_seen': slipped_entry_price
                    })
            active_positions.extend(new_positions)
            pending_orders = [] # 訂單執行完畢，清空待命區

            # ==========================================
            # 🎯 階段二：持倉出場判定 (嚴格先用「舊停損線」去防守今日下殺)
            # ==========================================
            still_active = []
            for pos in active_positions:
                exit_price = None
                exit_reason = ""

                # 🛑 優先判定是否觸發停損
                # ✅ 最短持倉保護期：進場後 N 個交易日內不允許停損出場
                #    避免正常波動在保護期內就被震出
                days_held = (index - pos['entry_date']).days if 'entry_date' in pos else 0
                in_protection = self.min_holding_days > 0 and days_held < self.min_holding_days

                if is_locked and (current_low <= pos['stop_loss']):
                    pass
                elif not in_protection and current_open <= pos['stop_loss'] * (1 - 0.005):
                    exit_slip = self._dynamic_slippage(current_open, self.slippage_pct)
                    exit_price = current_open * (1 - exit_slip)
                    # 移動停利模式下，若停損線已移到進場價以上，這是鎖利出場而非真停損
                    if self.bt_mode == "移動停利" and pos['stop_loss'] > pos['entry_price']:
                        exit_reason = "Trailing_Stop"
                    else:
                        exit_reason = "Gap_Through_Stop"
                elif not in_protection and current_low <= pos['stop_loss']:
                    exit_slip = self._dynamic_slippage(pos['stop_loss'], self.slippage_pct)
                    exit_price = pos['stop_loss'] * (1 - exit_slip)
                    # 移動停利模式下，若停損線已移到進場價以上，這是鎖利出場而非真停損
                    if self.bt_mode == "移動停利" and pos['stop_loss'] > pos['entry_price']:
                        exit_reason = "Trailing_Stop"
                    else:
                        exit_reason = "Hit_Stop"
                elif in_protection and current_low <= pos['stop_loss'] - pos['risk_dist']:
                    # 保護期內跌超過 2 倍停損距離（災難性下跌），用收盤價出場
                    # 不用最低價，因為恐慌盤中最低通常是當天最差點，收盤往往反彈
                    exit_slip = self._dynamic_slippage(current_close, self.slippage_pct)
                    exit_price = current_close * (1 - exit_slip)
                    exit_reason = "Catastrophic_Stop"
                elif self.bt_mode != "移動停利":
                    if current_open >= pos['take_profit']:
                        exit_slip = self._dynamic_slippage(current_open, self.slippage_pct)
                        exit_price = current_open * (1 - exit_slip)
                        exit_reason = "Gap_Over_Target"
                    elif current_high >= pos['take_profit']:
                        exit_slip = self._dynamic_slippage(pos['take_profit'], self.slippage_pct)
                        exit_price = pos['take_profit'] * (1 - exit_slip)
                        exit_reason = "Hit_Target"

                # 📊 基本面出場判定（每月第一個交易日檢查一次）
                # ✅ 套用最短持倉保護期：進場後 N 天內不做基本面出場（與停損一致）
                if exit_price is None and not in_protection and ticker and self.fundamental_cache:
                    current_month = (index.year, index.month)
                    if current_month != _last_fundamental_check_month:
                        _last_fundamental_check_month = current_month
                        fund_data = self.fundamental_cache.get(ticker, {})
                        rev_history    = fund_data.get("rev",    pd.DataFrame())
                        margin_history = fund_data.get("margin", pd.DataFrame())

                        # 用當時日期查詢歷史序列，完全無前視偏差
                        if not rev_history.empty or not margin_history.empty:
                            try:
                                from revenue_fetcher import get_fundamental_at_date
                                fund = get_fundamental_at_date(
                                    ticker, index,
                                    rev_history=rev_history,
                                    margin_history=margin_history,
                                )
                            except Exception:
                                fund = {}
                        else:
                            fund = {}

                        reasons = []
                        yoy = fund.get("latest_yoy")
                        if yoy is not None and yoy < 0:
                            reasons.append("YoY轉負")

                        gm_expanding = fund.get("gm_expanding")
                        if gm_expanding is not None and not gm_expanding:
                            reasons.append("毛利衰退")

                        if reasons:
                            fund_slip   = self._dynamic_slippage(current_close, self.slippage_pct)
                            exit_price  = current_close * (1 - fund_slip)
                            exit_reason = "Fundamental_Exit:" + "+".join(reasons)

                # ----------------- 結算交易 -----------------
                if exit_price is not None:
                    gross_exit_revenue = pos['shares'] * exit_price
                    
                    # 🛡️ 出場的 20 元手續費地板與證交稅
                    exit_fee = max(20.0, gross_exit_revenue * self.fee_rate)
                    exit_tax = gross_exit_revenue * self.tax_rate
                    
                    exit_revenue_net = gross_exit_revenue - exit_fee - exit_tax
                    net_profit = exit_revenue_net - pos['entry_cost']
                    cash += exit_revenue_net  # 結算回流現金池
                    
                    # 🛡️ 實戰重構 4：雙軌 R 系統 (策略 Alpha vs 現實摩擦力)
                    p_risk = pos['planned_risk_amount'] if pos['planned_risk_amount'] > 0 else 1.0
                    gross_r = (exit_price - pos['entry_price']) / (pos['entry_price'] - pos['sl_price_initial'])
                    real_net_r = net_profit / p_risk

                    trade_history.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': index,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'net_profit': net_profit,
                        'gross_r': gross_r,          # 純策略走勢 R 值
                        'real_r': real_net_r,        # 扣除稅費的現實 R 值
                        'reason': exit_reason,
                        'adjusted_risk': pos['risk_dist'],
                        'transaction_cost': exit_fee + exit_tax + (pos['entry_cost'] - (pos['shares'] * pos['entry_price']))
                    })
                else:
                    # ==========================================
                    # 🎯 階段三：移動停利更新 (如果今天活下來了，才用今天的最高價更新明日停損)
                    # 徹底消除「用下午高點保護早盤下殺」的未來函數
                    # ==========================================
                    if self.bt_mode == "移動停利":
                        if current_high > pos.get('highest_seen', pos['entry_price']):
                            pos['highest_seen'] = current_high
                            # ── 問題 3 修正：至少達到 1R 才啟動移動停利 ──
                            min_activate_price = pos['entry_price'] + pos['risk_dist']
                            if pos['highest_seen'] >= min_activate_price:
                                new_stop = pos['highest_seen'] - (pos['risk_dist'] * self.rr_ratio)
                                if new_stop > pos['stop_loss']:
                                    pos['stop_loss'] = new_stop
                    
                    still_active.append(pos)
            
            active_positions = still_active

            # ==========================================
            # 🎯 階段四：計算日終淨值 & 判斷今日收盤訊號
            # ==========================================
            current_mtm_equity = cash + sum([p['shares'] * current_close for p in active_positions])
            mtm_equity_curve.append(current_mtm_equity)

            fvg = getattr(row, 'FVG', False)
            adj_risk = getattr(row, 'adjusted_risk', 0)

            # ── 問題 1 修正：用張數（Volume）而非金額判斷流動性 ──
            current_volume = getattr(row, 'Volume', 0)
            vol_ma20_val   = getattr(row, 'Vol_SMA20', current_volume)  # 20日均量
            liquidity_ok   = (vol_ma20_val >= self.min_vol) if self.min_vol > 0 else True

            # ── 問題 2 修正：FVG 確認條件（缺口未被回補才進場）──
            # 今天收盤要在前天高點以上（缺口下緣）
            prev_high = getattr(row, 'Prev_High', 0)  # 在 apply_smc_logic 已計算
            fvg_confirmed = (current_close >= prev_high) if prev_high > 0 else True

            # 今天收盤產生訊號，不立刻買入！而是推入「待命區」，明天開盤才執行
            if fvg and adj_risk > 0 and liquidity_ok and fvg_confirmed:
                # ── 進場基本面過濾（YoY / 毛利率擴張，無前視偏差）──
                fundamental_ok = True
                if (self.entry_min_yoy is not None or self.entry_require_gm or self.entry_require_ocf or self.entry_max_debt_ratio is not None) and ticker and self.fundamental_cache:
                    fund_data = self.fundamental_cache.get(ticker, {})
                    rev_h = fund_data.get("rev", pd.DataFrame())
                    mar_h = fund_data.get("margin", pd.DataFrame())
                    cf_h  = fund_data.get("cashflow", pd.DataFrame())
                    dr_h  = fund_data.get("debt", pd.DataFrame())
                    if not rev_h.empty or not mar_h.empty or not cf_h.empty or not dr_h.empty:
                        try:
                            from revenue_fetcher import get_fundamental_at_date
                            f = get_fundamental_at_date(ticker, index, rev_history=rev_h, margin_history=mar_h, cashflow_history=cf_h, debt_history=dr_h)
                            # YoY 門檻
                            if self.entry_min_yoy is not None:
                                yoy = f.get("latest_yoy")
                                if yoy is None or yoy < self.entry_min_yoy:
                                    fundamental_ok = False
                            # 毛利率擴張
                            if self.entry_require_gm and fundamental_ok:
                                gm_e = f.get("gm_expanding")
                                if gm_e is not None and not gm_e:
                                    fundamental_ok = False
                            # 營業現金流為正（抓不到資料不誤殺，與掃描端一致）
                            if self.entry_require_ocf and fundamental_ok:
                                ocf_pos = f.get("ocf_positive")
                                if ocf_pos is not None and not ocf_pos:
                                    fundamental_ok = False
                            # 負債比率上限（抓不到資料不誤殺）
                            if self.entry_max_debt_ratio is not None and fundamental_ok:
                                dr = f.get("debt_ratio")
                                if dr is not None and dr > self.entry_max_debt_ratio:
                                    fundamental_ok = False
                        except Exception:
                            pass  # 查詢失敗不擋，避免誤殺

                if fundamental_ok and adj_risk > 1e-5:
                    pending_orders.append({'risk_dist': adj_risk})

        # 迴圈結束後，結算 DataFrame 指標
        df['MtM_Equity'] = mtm_equity_curve
        df['Daily_Return'] = df['MtM_Equity'].pct_change().fillna(0)
        
        daily_volatility = df['Daily_Return'].std()
        sharpe_ratio = (df['Daily_Return'].mean() / daily_volatility) * np.sqrt(252) if daily_volatility > 0 else 0.0
            
        trades_df = pd.DataFrame(trade_history)
        return trades_df, df, sharpe_ratio


# 共用欄位對照表，避免兩條執行路徑各自維護
_TRADE_RENAME_MAP = {
    'entry_time': '日期',
    'exit_time': '出場日期',
    'entry_price': '進場',
    'exit_price': '出場',
    'real_r': '真實R',
    'gross_r': '走勢R',
    'net_profit': '絕對損益',
    'adjusted_risk': '每股風險',
}

def _rename_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """統一套用欄位中文化，兩條執行路徑共用"""
    return trades_df.rename(columns=_TRADE_RENAME_MAP)


# 🌟 完美相容轉接器 (完全不用改外層 portfolio_engine 和 app.py)
def run_backtest_core(df, bt_mode="固定盈虧比", bt_val=3.0, slippage_pct=0.002, fee_disc=DEFAULT_FEE_DISC, tax_rate=TAX_RATE, min_vol=0, ticker="", fundamental_cache=None, min_holding_days=0, **kwargs):
    actual_fee_rate = FEE_RATE * fee_disc

    if bt_mode == "R倍數":
        bt_mode = "固定盈虧比"

    engine = InstitutionalBacktestEngine(
        initial_capital=1000000.0,
        risk_per_trade=0.02,
        fee_rate=actual_fee_rate,
        tax_rate=tax_rate,
        slippage_pct=slippage_pct,
        rr_ratio=bt_val,
        bt_mode=bt_mode,
        fundamental_cache=fundamental_cache or {},
        min_holding_days=min_holding_days,
        min_vol=min_vol,
        entry_min_yoy=kwargs.get('entry_min_yoy', None),
        entry_require_gm=kwargs.get('entry_require_gm', False),
        entry_require_ocf=kwargs.get('entry_require_ocf', False),
        entry_max_debt_ratio=kwargs.get('entry_max_debt_ratio', None),
    )
    
    trades_df, updated_df, sharpe_ratio = engine.run_backtest(df, ticker=ticker)
    
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(), updated_df
    
    compat_trades_df = _rename_trades(trades_df)
    compat_trades_df['每股風險'] = compat_trades_df.get('每股風險', compat_trades_df['進場'] * 0.05)
    return compat_trades_df, updated_df