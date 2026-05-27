# SMC-Alpha-Portfolio-Engine
基於 Python 開發的 SMC 量化投資組合回測引擎，包含動態風控與蒙地卡羅壓力測試。
這是一個獨立使用 Python 開發的投資組合級別量化回測引擎，專注於高度擬真的交易執行環境與動態風險控管。

## 核心功能與技術特點
* **高擬真交易執行 (Execution Realism):** 實作次條圖執行邏輯（Next-bar execution）、流動性限制與投資組合資金動態配置，確保回測結果貼近真實市場。
* **動態風險控管 (Risk Management):** 內建基於市場波動率的減倉風控機制（Volatility-based risk reduction），並整合蒙地卡羅壓力測試（Monte Carlo Simulation）以驗證策略極端狀況下的魯棒性。
* **效能優化 (Performance Optimization):** 利用多進程（Multiprocessing）與向量化計算（Vectorized calculations）優化大規模數據模擬，大幅提升回測速度。

## 開發工具
* Python (Pandas / NumPy)
* 數據視覺化 (Plotly / Streamlit)
