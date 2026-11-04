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

    # 大盤：優先 FinMind 的 TAIEX 指數，取不到就退回 0050（與既有 SMC 系統一致）
    benchmark_id: str = "TAIEX"
    benchmark_fallback_id: str = "0050"

    # ---- 個股過濾（全部「逐日、point-in-time」判斷，避免倖存者 / 前視偏差）----
    min_listed_days: int = 250            # 截至當日的交易日數 >= 250 才納入
    turnover_window: int = 20             # 日均成交值的計算窗
    min_avg_turnover: float = 20_000_000  # 日均成交值門檻（新台幣）
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
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    sector: SectorConfig = field(default_factory=SectorConfig)


DEFAULT_CONFIG = Config()
