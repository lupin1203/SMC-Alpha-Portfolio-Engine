"""簡單趨勢策略: 通道突破 (Donchian) + ATR 初始停損 + 反向通道移動出場。

- 收盤突破過去 entry_n 根最高價 → 下一根開盤做多 (both 模式下, 跌破最低價做空)
- 初始停損: 進場價 - atr_mult * ATR(20)
- 出場: 收盤跌破過去 exit_n 根最低價 → 下一根開盤平倉; 或盤中碰到停損
- 成本: 進出場市價 taker 0.05% + 滑價 0.02%; 資金費率: 多單每 8 小時付 0.01%, 空單收 0.01%
- 報酬以 R (初始風險) 計算
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smc_engine import Trade


@dataclass
class TrendParams:
    entry_n: int = 20
    exit_n: int = 10
    atr_mult: float = 2.0
    both: bool = False  # False = 只做多
    fee: float = 0.0005
    slippage: float = 0.0002
    funding_8h: float = 0.0001


def atr(df, n=20):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy()


def backtest_trend(df: pd.DataFrame, p: TrendParams) -> list[Trade]:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(df)
    hh = df["high"].rolling(p.entry_n).max().shift(1).to_numpy()
    ll = df["low"].rolling(p.entry_n).min().shift(1).to_numpy()
    xh = df["high"].rolling(p.exit_n).max().shift(1).to_numpy()
    xl = df["low"].rolling(p.exit_n).min().shift(1).to_numpy()
    bar_hours = (df.index[1] - df.index[0]).total_seconds() / 3600
    trades, pos, pending, pending_exit = [], None, 0, False
    cost = p.fee + p.slippage

    def close(t, i, px):
        t.exit_i, t.exit = i, px
        risk = abs(t.entry - t.sl0)
        move = (px - t.entry) * t.dir
        held_h = (i - t.entry_i) * bar_hours
        funding = t.entry * p.funding_8h * held_h / 8 * t.dir  # 多單付, 空單收
        t.r_gross = move / risk
        t.r_net = (move - t.entry * cost - px * cost - funding) / risk
        trades.append(t)

    for i in range(len(c)):
        # 開盤: 執行上一根收盤產生的訊號
        if pos is not None and pending_exit:
            close(pos, i, o[i]); pos, pending_exit = None, False
        if pos is None and pending != 0 and not np.isnan(a[i - 1]):
            d = pending
            sl = o[i] - d * p.atr_mult * a[i - 1]
            pos = Trade(d, i, o[i], sl, np.nan)
            pos.sl0 = sl
        pending = 0
        # 盤中停損
        if pos is not None and ((l[i] <= pos.sl) if pos.dir == 1 else (h[i] >= pos.sl)):
            px = min(o[i], pos.sl) if pos.dir == 1 else max(o[i], pos.sl)
            close(pos, i, px); pos = None
        # 收盤: 產生訊號
        if np.isnan(hh[i]) or np.isnan(xl[i]):
            continue
        if pos is None:
            if c[i] > hh[i]:
                pending = 1
            elif p.both and c[i] < ll[i]:
                pending = -1
        elif (pos.dir == 1 and c[i] < xl[i]) or (pos.dir == -1 and c[i] > xh[i]):
            pending_exit = True
    return trades


def equity_stats(df, trades, risk_pct=0.01):
    """每筆冒 risk_pct 風險 (複利) 的年化報酬與最大回撤。"""
    eq, peak, mdd = 1.0, 1.0, 0.0
    for t in trades:
        eq *= 1 + risk_pct * t.r_net
        peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = eq ** (1 / years) - 1 if eq > 0 else -1
    return cagr, mdd
