# SMC 回測 (OKX 永續)

與 `tradingview/smc_auto_structure.pine` 同一套規則的 Python 版本與回測。

```bash
python fetch_okx.py BTC-USDT-SWAP 15m 730   # 下載 2 年 15m K線到 data/ (公開 API, 不需 Key)
python fetch_okx.py ETH-USDT-SWAP 15m 730
python run_backtest.py                      # 參數掃描 → results/grid_results.csv
```

- `smc_engine.py`：結構偵測 (swing / CHoCH / OB)、高週期過濾、逐K回測
- 成本：限價進場與停利 maker 0.02%，停損 taker 0.05% + 0.02% 滑價
- 同一根K棒同時碰到停損與停利 → 算停損；成交當根就碰到停損 → 算停損
- 前一年為樣本內、後一年為樣本外 (`period` 欄位: all / in / out)
