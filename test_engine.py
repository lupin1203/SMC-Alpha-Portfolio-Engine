# test_engine.py
import unittest
import pandas as pd
from strategy import SMCStrategy

class TestSMCStrategy(unittest.TestCase):
    def setUp(self):
        # 建立假數據模擬完美缺口
        data = {
            'Open': [100, 105, 102, 110],
            'High': [102, 106, 104, 115],
            'Low':  [98,  103, 101, 108],
            'Close':[101, 104, 103, 112]
        }
        self.df = pd.DataFrame(data)
        self.strategy = SMCStrategy()

    def test_fvg_detection(self):
        # 第4根Low(108) > 第2根High(106)，產生FVG
        res_df = self.strategy.generate_signals(self.df, min_gap_pct=0.1, use_sma_filter=False)
        self.assertFalse(res_df['FVG'].iloc[2], "第三根不該有缺口")
        self.assertTrue(res_df['FVG'].iloc[3], "第四根必須判定出缺口")
        print("✅ 單元測試通過：FVG 缺口判定邏輯正常！")

if __name__ == '__main__':
    unittest.main()