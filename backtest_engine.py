# backtest_engine.py
import numpy as np
import pandas as pd
import logging

# 台股漲跌幅限制 10%。資料為還原權息價，除權息日附近比例會有些微誤差，
# 因此用 9.5% 當門檻判定「漲停 / 跌停」。
LIMIT_THRESHOLD = 0.095

# ==========================================
# 回測真實度選項
# 預設值 = 原始邏輯 (完全不改變舊版結果)；REALISTIC_PRESET = 模擬實盤真的做得到的事
# ==========================================
REALISM_DEFAULTS = {
    # 進場方式：
    #   "close"     = 訊號當天收盤價進場 (原始邏輯，實盤其實做不到)
    #   "next_stop" = 隔天掛觸價單 (觸價 = 訊號日收盤)，盤中最高價 ≥ 觸價才成交，
    #                 成交價 = max(開盤, 觸價)；沒觸發就取消
    #   "retest"    = 回檔掛單：隔天起在缺口上緣 (訊號日最低價) 掛限價買單，retest_days 天內有效。
    #                 盤中最低價 ≤ 限價才成交，成交價 = min(開盤, 限價)；
    #                 開盤已跌破停損價則取消 (不會明知會停損還買)，過期也取消
    "entry_mode": "close",
    "retest_days": 5,
    # 漲停買不到：close 模式下訊號日收盤漲停則跳過；next_stop 模式下成交日一字漲停則跳過
    "skip_limit_up": False,
    # 跌停賣不掉：一字跌停 (最高 = 最低 且跌幅達跌停) 當天無法出場，留到隔天再判斷
    "limit_down_lock": False,
    # 保護期：進場後 N 個交易日內只用「進場價 − emergency_mult × 風險距」的緊急停損 (0 = 關閉)
    "protection_days": 0,
    "emergency_mult": 2.0,
    # 同一檔已有部位 (或掛單中) 時，略過新訊號
    "one_position_per_ticker": False,
    # K 棒內順序：先用昨天的停損判斷今天是否出場，收盤後才用今天最高價更新明天的移動停利
    # (原始邏輯是先用今天最高價上調停損，再用今天最低價判斷，隱含偷看順序)
    "realistic_intrabar": False,
    # 資料結束時仍持有的部位，以最後收盤價結算 (原始邏輯會直接丟掉，造成未結算部位消失)
    "close_open_at_end": False,
}

REALISTIC_PRESET = {
    "entry_mode": "next_stop",
    "skip_limit_up": True,
    "limit_down_lock": True,
    "protection_days": 10,
    "emergency_mult": 2.0,
    "one_position_per_ticker": True,
    "realistic_intrabar": True,
    "close_open_at_end": True,
}


def resolve_realism(realistic=False, **overrides):
    """組合真實度選項：先取預設或實盤預設，再套用個別覆寫 (只認得 REALISM_DEFAULTS 內的鍵)。"""
    opts = dict(REALISTIC_PRESET if realistic else REALISM_DEFAULTS)
    for k, v in overrides.items():
        if k in REALISM_DEFAULTS and v is not None:
            opts[k] = v
    return opts


class InstitutionalBacktestEngine:
    def __init__(self, initial_capital=1000000.0, risk_per_trade=0.02,
                 fee_rate=0.001425, tax_rate=0.003, slippage_pct=0.002, rr_ratio=3.0, bt_mode="固定盈虧比",
                 **realism):
        self.initial_capital = initial_capital
        self.max_risk_pct = risk_per_trade
        self.fee_rate = fee_rate
        self.tax_rate = tax_rate
        self.slippage_pct = slippage_pct
        self.rr_ratio = rr_ratio
        self.bt_mode = bt_mode  # 出場模式
        self.opts = resolve_realism(**realism)

    # ------------------------------------------
    # 內部工具
    # ------------------------------------------
    def _close_position(self, pos, index, exit_price, exit_reason):
        exit_revenue = pos['shares'] * exit_price * (1 - self.fee_rate - self.tax_rate)
        net_profit = exit_revenue - pos['entry_cost']
        p_risk = pos['planned_risk_amount'] if pos['planned_risk_amount'] > 0 else 1.0
        record = {
            'entry_time': pos['entry_time'],
            'exit_time': index,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'shares': pos['shares'],
            'net_profit': net_profit,
            'real_r': net_profit / p_risk,
            'reason': exit_reason,
            'adjusted_risk': pos['risk_dist'],
            'signal_time': pos['signal_time'],
        }
        return exit_revenue, record

    def _open_position(self, index, signal_time, entry_price, stop_price, risk_dist, planned_risk_amount, shares):
        entry_cost = shares * entry_price * (1 + self.fee_rate + self.slippage_pct)
        pos = {
            'entry_time': index,
            'signal_time': signal_time,
            'entry_price': entry_price,
            'stop_loss': stop_price,
            'initial_stop': stop_price,
            'take_profit': entry_price + (risk_dist * self.rr_ratio),
            'shares': shares,
            'entry_cost': entry_cost,
            'planned_risk_amount': planned_risk_amount,
            'risk_dist': risk_dist,
            'highest_seen': entry_price,  # 供移動停利用
            'bars_held': 0,
        }
        return pos, entry_cost

    def _size(self, current_mtm_equity, risk_dist):
        # 確保就算單股帳戶破產，也用初始資金基準去計算部位，不漏掉任何給外層引擎的訊號
        virtual_equity = max(current_mtm_equity, self.initial_capital)
        planned_risk_amount = virtual_equity * self.max_risk_pct
        shares = int(planned_risk_amount / risk_dist)
        # 強制至少 1 股，確保真實 R 倍數能被計算並傳遞給 portfolio_engine
        if shares <= 0:
            shares = 1
            planned_risk_amount = risk_dist
        return shares, planned_risk_amount

    def _update_trailing(self, pos, high):
        if self.bt_mode == "移動停利" and high > pos.get('highest_seen', pos['entry_price']):
            pos['highest_seen'] = high
            # 根據最高價，向下回落 (風險距 * 設定的倍數) 作為移動停損點
            new_stop = pos['highest_seen'] - (pos['risk_dist'] * self.rr_ratio)
            if new_stop > pos['stop_loss']:
                pos['stop_loss'] = new_stop

    def _effective_stop(self, pos):
        o = self.opts
        if o['protection_days'] > 0 and pos['bars_held'] < o['protection_days']:
            # 保護期：只用緊急停損；但若移動停利已經上調到初始停損之上，就以移動停利為準
            if pos['stop_loss'] > pos['initial_stop']:
                return pos['stop_loss'], "Hit_Stop"
            return pos['entry_price'] - pos['risk_dist'] * o['emergency_mult'], "Hit_Emergency_Stop"
        return pos['stop_loss'], "Hit_Stop"

    # ------------------------------------------
    # 主迴圈
    # ------------------------------------------
    def run_backtest(self, df):
        o = self.opts
        cash = self.initial_capital
        active_positions = []
        pending_order = None   # next_stop 模式的隔日觸價單
        trade_history = []
        mtm_equity_curve = []

        if df is None or df.empty or 'Close' not in df.columns:
            return pd.DataFrame(), df, 0.0

        has_vol = 'Turnover_5MA' in df.columns
        prev_close = None
        current_mtm_equity = cash
        index = None
        current_close = None

        # 🚀 效能優化：改用 itertuples 進行迭代
        for row in df.itertuples():
            index = row.Index
            current_open = row.Open
            current_high = row.High
            current_low = row.Low
            current_close = row.Close

            has_prev = prev_close is not None and prev_close > 0
            locked_limit_up = has_prev and current_high == current_low and current_close >= prev_close * (1 + LIMIT_THRESHOLD)
            locked_limit_down = has_prev and current_high == current_low and current_close <= prev_close * (1 - LIMIT_THRESHOLD)

            # ==========================================
            # 📥 隔日觸價單成交判定 (next_stop 模式)
            # ==========================================
            if pending_order is not None and pending_order['type'] == 'limit':
                order = pending_order
                limit = order['limit']
                stop_price = limit - order['risk_dist']
                if current_open <= stop_price:
                    pending_order = None  # 開盤已跌破停損，取消掛單
                elif current_low <= limit:
                    pending_order = None
                    fill_price = min(current_open, limit)
                    pos, entry_cost = self._open_position(
                        index, order['signal_time'], fill_price, stop_price,
                        order['risk_dist'], order['planned_risk_amount'], order['shares'])
                    cash -= entry_cost
                    active_positions.append(pos)
                else:
                    order['days_left'] -= 1
                    if order['days_left'] <= 0:
                        pending_order = None  # 過期
            elif pending_order is not None:
                order, pending_order = pending_order, None
                trigger = order['trigger']
                if o['skip_limit_up'] and locked_limit_up:
                    pass  # 一字漲停，排隊買不到
                elif current_high >= trigger:
                    fill_price = max(current_open, trigger)
                    # 停損價在掛單時就決定 (觸價 − 風險距)；跳空高開成交時實際風險會變大，R 也會如實反映
                    pos, entry_cost = self._open_position(
                        index, order['signal_time'], fill_price, trigger - order['risk_dist'],
                        order['risk_dist'], order['planned_risk_amount'], order['shares'])
                    cash -= entry_cost
                    active_positions.append(pos)

            # ==========================================
            # 🎯 出場判定
            # ==========================================
            still_active = []
            for pos in active_positions:
                exit_price = None
                exit_reason = ""

                if not o['realistic_intrabar']:
                    # 原始邏輯：先用今天最高價上調移動停利，再判斷
                    self._update_trailing(pos, current_high)

                stop_price, stop_reason = self._effective_stop(pos)

                if o['limit_down_lock'] and locked_limit_down:
                    pass  # 一字跌停，賣不掉
                elif current_open <= stop_price:
                    exit_price = current_open * (1 - self.slippage_pct)  # 跳空跌破，開盤即砍
                    exit_reason = "Gap_Through_Stop"
                elif current_low <= stop_price:
                    exit_price = stop_price * (1 - self.slippage_pct)
                    exit_reason = stop_reason
                elif self.bt_mode != "移動停利":
                    # 固定盈虧比模式
                    if current_open >= pos['take_profit']:
                        exit_price = current_open * (1 - self.slippage_pct)
                        exit_reason = "Gap_Over_Target"
                    elif current_high >= pos['take_profit']:
                        exit_price = pos['take_profit'] * (1 - self.slippage_pct)
                        exit_reason = "Hit_Target"

                if exit_price is not None:
                    exit_revenue, record = self._close_position(pos, index, exit_price, exit_reason)
                    cash += exit_revenue
                    trade_history.append(record)
                else:
                    if o['realistic_intrabar']:
                        # 收盤後才用今天最高價更新明天的移動停利
                        self._update_trailing(pos, current_high)
                    pos['bars_held'] += 1
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

            prev_close_for_signal = prev_close
            prev_close = current_close

            # ==========================================
            # 🌟 進場邏輯
            # ==========================================
            if not (fvg and adj_risk > 0 and liquidity_ok):
                continue
            risk_dist = adj_risk
            if risk_dist <= 1e-5:
                continue
            if o['one_position_per_ticker'] and (active_positions or pending_order is not None):
                continue

            shares, planned_risk_amount = self._size(current_mtm_equity, risk_dist)

            if o['entry_mode'] == "retest":
                # 新訊號取代舊的未成交掛單
                pending_order = {
                    'type': 'limit',
                    'signal_time': index,
                    'limit': getattr(row, 'fvg_top', current_close),
                    'risk_dist': risk_dist,
                    'shares': shares,
                    'planned_risk_amount': planned_risk_amount,
                    'days_left': int(o['retest_days']),
                }
            elif o['entry_mode'] == "next_stop":
                pending_order = {
                    'type': 'stop',
                    'signal_time': index,
                    'trigger': current_close,
                    'risk_dist': risk_dist,
                    'shares': shares,
                    'planned_risk_amount': planned_risk_amount,
                }
            else:
                if (o['skip_limit_up'] and prev_close_for_signal
                        and current_close >= prev_close_for_signal * (1 + LIMIT_THRESHOLD)):
                    continue  # 收盤漲停，收盤價買不到
                entry_price = current_close
                pos, entry_cost = self._open_position(
                    index, index, entry_price, entry_price - risk_dist, risk_dist, planned_risk_amount, shares)
                cash -= entry_cost  # 允許內部 cash 變負數，我們只在乎 R 倍數與訊號生命週期
                active_positions.append(pos)

        # ==========================================
        # 資料結束：結算仍持有的部位
        # ==========================================
        if o['close_open_at_end'] and active_positions:
            for pos in active_positions:
                exit_revenue, record = self._close_position(
                    pos, index, current_close * (1 - self.slippage_pct), "End_Of_Data")
                cash += exit_revenue
                trade_history.append(record)
            mtm_equity_curve[-1] = cash

        df['MtM_Equity'] = mtm_equity_curve
        df['Daily_Return'] = df['MtM_Equity'].pct_change().fillna(0)

        daily_volatility = df['Daily_Return'].std()
        sharpe_ratio = (df['Daily_Return'].mean() / daily_volatility) * np.sqrt(252) if daily_volatility > 0 else 0.0

        trades_df = pd.DataFrame(trade_history)
        return trades_df, df, sharpe_ratio


# 🌟 相容轉接器 (相容組合引擎的呼叫)
def run_backtest_core(df, bt_mode="固定盈虧比", bt_val=3.0, slippage_pct=0.002, fee_disc=0.28, tax_rate=0.003, min_vol=0,
                      realistic=False, **kwargs):
    """
    realistic=True 套用 REALISTIC_PRESET；也可用 kwargs 個別覆寫 REALISM_DEFAULTS 中的選項
    (例如 protection_days=0)。不認得的 kwargs 會被忽略，維持舊版呼叫相容。
    """
    actual_fee_rate = 0.001425 * fee_disc

    # 向下相容舊版參數名
    if bt_mode == "R倍數":
        bt_mode = "固定盈虧比"

    realism = resolve_realism(realistic=realistic, **kwargs)
    engine = InstitutionalBacktestEngine(
        initial_capital=1000000.0,
        risk_per_trade=0.02,
        fee_rate=actual_fee_rate,
        tax_rate=tax_rate,
        slippage_pct=slippage_pct,
        rr_ratio=bt_val,
        bt_mode=bt_mode,
        **realism
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
        'net_profit': '絕對損益',
        'signal_time': '訊號日',
    })

    compat_trades_df['每股風險'] = compat_trades_df['adjusted_risk']
    return compat_trades_df, updated_df
