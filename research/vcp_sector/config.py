"""參數集中管理。

原則：參數先固定、不做優化。之後的 walk-forward 只允許在 in-sample 區間挑參數，
所以這裡的預設值就是「事前寫死」的版本，請勿依回測結果回頭改。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]


@dataclass(frozen=True)
class DataConfig:
    # 既有 SMC 系統的 K 線快取（data_provider.py / update_data.py 寫入）
    local_data_dir: Path = REPO_ROOT / "local_data"
    # 本模組自己從 FinMind 抓下來的快取
    cache_dir: Path = MODULE_DIR / "cache"
    stock_list_csv: Path = REPO_ROOT / "stock_list.csv"
    finmind_url: str = "https://api.finmindtrade.com/api/v4/data"
    finmind_token_env: str = "FINMIND_TOKEN"
    start_date: str = "2015-01-01"
    yf_batch_size: int = 100
    session_close: str = "14:30"   # 台北時間此時之後才視為今日 K 棒完成（收盤 13:30，留緩衝）
    max_stale_bdays: int = 3       # 本模組快取落後超過 N 個營業日就重新下載

    # 大盤：優先 FinMind 的 TAIEX 指數，取不到就退回 0050（與既有 SMC 系統一致）
    benchmark_id: str = "TAIEX"
    benchmark_fallback_id: str = "0050"

    # ---- 個股過濾（全部「逐日、point-in-time」判斷，避免倖存者 / 前視偏差）----
    min_listed_days: int = 250            # 截至當日的交易日數 >= 250 才納入
    turnover_window: int = 20             # 日均成交值的計算窗
    min_avg_turnover: float = 20_000_000  # 日均成交值門檻（新台幣）
    excluded_types: tuple = ("emerging",)  # FinMind TaiwanStockInfo.type：twse / tpex / emerging
    # FinMind TaiwanStockInfo 中要剔除的 industry_category
    excluded_categories: tuple = (
        "ETF", "ETN", "上櫃指數股票型基金(ETF)", "指數投資證券(ETN)",
        "受益證券", "存託憑證", "Index", "大盤", "所有證券", "創新版股票",
    )


@dataclass(frozen=True)
class SectorConfig:
    groups_yaml: Path = MODULE_DIR / "groups.yaml"
    min_members: int = 4          # 當日有效成員 < 4 的族群不計算

    ma_short: int = 20
    ma_long: int = 50
    new_high_window: int = 20
    ret_window: int = 20

    # 族群轉強定義（簡單版）
    strong_pct_above_ma20: float = 0.6
    strong_rank: float = 0.7
    strong_breadth_lookback: int = 5   # pct_above_ma20 需比 N 日前上升


@dataclass(frozen=True)
class LeadLagConfig:
    ret_window: int = 20
    n_leaders: int = 2               # 龍頭：近 20 日報酬最高的 1–2 檔
    leader_turnover_pct: float = 0.5 # 且成交值在族群內排前 50%
    n_laggards: int = 3              # 落後：報酬 < 族群中位數、close > MA50，取最接近中位數的 3 檔
    ma_long: int = 50


@dataclass(frozen=True)
class VCPConfig:
    """VCP 辨識參數（事先固定）。"""
    min_swing_pct: float = 0.04       # ZigZag 反轉門檻
    # 1. 前置趨勢（Trend Template 簡化版）
    ma_fast: int = 50
    ma_mid: int = 150
    ma_slow: int = 200
    ma_slow_rising_days: int = 20     # MA200 需高於 20 日前
    max_below_52w_high: float = 0.25  # 距 52 週高點 <= 25%
    min_above_52w_low: float = 0.30   # 距 52 週低點 >= 30%
    # 2. 收縮結構
    lookback: int = 120               # 整理起點（最高 swing high）需在近 120 日內
    min_contractions: int = 2
    max_contractions: int = 4
    contraction_ratio: float = 0.8    # 每段深度 <= 前一段 × 0.8
    max_first_depth: float = 0.35
    max_last_depth: float = 0.10
    # 人工檢查 30 例後加入（依看圖判斷決定，非依回測結果）：
    min_base_days: int = 20           # 整理期至少約一個月的交易日
    max_pivot_below_base_high: float = 0.10  # pivot 需在整理區最高點 10% 以內，否則只是區間內反彈
    max_chase_pct: float = 0.05       # 隔日開盤 > pivot × 1.05 不追（使用者決定）
    # 3. 量縮
    vol_dry_ratio: float = 0.7        # 最後一段均量 < 整理前 50 日均量 × 0.7
    pre_base_vol_days: int = 50
    # 5. 突破
    breakout_vol_mult: float = 1.5    # 當日量 > 前 50 日均量 × 1.5
    breakout_vol_days: int = 50


@dataclass(frozen=True)
class TradeConfig:
    """進出場規則：事先固定，不做優化。"""
    stop_lookback: int = 10          # 停損 = 訊號日（含）前 10 日最低價（訊號 B 用；VCP 會用自己的 stop）
    max_risk_pct: float = 0.15       # 進場價到停損距離 > 15% 的不做（R 太大、部位太小）
    min_risk_pct: float = 0.01       # < 1% 的停損太近，R 會被雜訊放大，也不做
    trail_pct: float = 0.10          # 移動停利：收盤 < 進場後最高收盤 × (1 − 10%) → 隔日開盤出
    exit_ma: int = 20                # 收盤由上往下跌破 MA20 → 隔日開盤出
    max_hold_days: int = 120         # 時間停損
    fee_rate: float = 0.001425       # 與 SMC 系統 config.py 相同
    tax_rate: float = 0.003
    slippage: float = 0.001          # 進出各 0.1%

    # 投組模擬
    risk_per_trade: float = 0.01     # 每筆風險 = 帳戶 1%
    max_positions: int = 10
    max_position_pct: float = 0.20   # 單一部位上限 = 帳戶 20%
    initial_capital: float = 1_000_000.0
    lot_size: int = 1                # 1 = 可零股（2020 起盤中零股）；1000 = 只能整張（會偏向低價股）


@dataclass(frozen=True)
class WalkForwardConfig:
    is_years: int = 3
    oos_years: int = 1
    min_is_trades: int = 30          # IS 交易數不足的參數組不選
    # 敏感度 / walk-forward 的小範圍網格（訊號 B：族群門檻）
    grid_pct_above_ma20: tuple = (0.5, 0.6, 0.7)
    grid_rank: tuple = (0.6, 0.7, 0.8)
    # 訊號 A / C：VCP 參數
    grid_min_swing_pct: tuple = (0.03, 0.04, 0.05)
    grid_vol_dry_ratio: tuple = (0.6, 0.7, 0.8)
    n_random_baseline: int = 20


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    sector: SectorConfig = field(default_factory=SectorConfig)
    leadlag: LeadLagConfig = field(default_factory=LeadLagConfig)
    vcp: VCPConfig = field(default_factory=VCPConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    wf: WalkForwardConfig = field(default_factory=WalkForwardConfig)


DEFAULT_CONFIG = Config()
