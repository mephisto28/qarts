import unittest
import numpy as np
import pandas as pd

from qarts.core.panel import PanelBlockIndexed, PanelBlockDense
from qarts.core.convert_utils import build_ranges, make_time_grid, densify_features_from_df

class TestPanelConversion(unittest.TestCase):

    def setUp(self):
        # 1. 构造测试数据
        # 时间范围：4天
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        
        # 两个标的：InstA, InstB, InstC
        # InstA: 第2天(dates[1]) 缺失数据
        # InstB: 第3天(dates[2]) 缺失数据
        # InstC: 第1天(dates[0]) 缺失数据

        data = {
            'datetime': [
                dates[0], dates[2], dates[3], dates[4], # InstA 缺少 dates[1]
                dates[0], dates[1], dates[3], dates[4], # InstB 缺少 dates[2]
                dates[1], dates[2], dates[3], dates[4]  # InstC 缺少 dates[0]
            ],
            'instrument': [
                'InstA', 'InstA', 'InstA', 'InstA',
                'InstB', 'InstB', 'InstB', 'InstB',
                'InstC', 'InstC', 'InstC', 'InstC'
            ],
            'price': [
                100.0, 102.0, 103.0, 104.0, # InstA
                200.0, 201.0, 203.0, 204.0, # InstB
                300.0, 301.0, 303.0, 304.0, # InstC
            ],
            'volume': [
                1000, 1200, 1300, 1400, # InstA
                2000, 2100, 2300, 2400, # InstB
                3000, 3100, 3300, 3400, # InstC
            ]
        }
        
        df = pd.DataFrame(data)
        self.raw_df = df.set_index(['datetime', 'instrument']).sort_index()
        self.block_indexed = PanelBlockIndexed(self.raw_df, order='datetime-first')
        self.all_timestamps = dates.values
        self.all_instruments = np.array(['InstA', 'InstB', 'InstC'])

    def test_indexed_to_dense_conversion(self):
        """
        测试从 Indexed Block 到 Dense Block 的转换，
        重点验证 'price' (ffill) 和 'volume' (fill 0) 的行为。
        """
        # fill_methods: 1 for price, 0 for volume
        dense_block = PanelBlockDense.from_indexed_block(
            self.block_indexed,
            required_columns=['price', 'volume'],
            fill_methods=[1, 0], 
            frequency='1D',
            inst_cats=self.all_instruments
        )

        # --- 基础维度检查 ---
        self.assertEqual(dense_block.data.shape, (2, 3, 5)) # (Fields, Insts, Time)
        self.assertTrue(np.array_equal(dense_block.fields, ['price', 'volume']))
        self.assertTrue(np.array_equal(dense_block.instruments, ['InstA', 'InstB', 'InstC']))

        # --- 获取视图 ---
        price_view = dense_block.get_view('price')   # Shape: (2, 4) -> (Inst, Time)
        vol_view = dense_block.get_view('volume')    # Shape: (2, 4)

        # --- 验证 1：价格 Forward Fill (InstA) ---
        # InstA (index 0) 在 Day 1 (index 1) 是缺失的
        # Day 0: 100.0 -> Day 1: 应该也是 100.0 (ffill)
        self.assertEqual(price_view[0, 0], 100.0) # T0
        self.assertEqual(price_view[0, 1], 100.0) # T1 (Filled)
        self.assertEqual(price_view[0, 2], 102.0) # T2 (Real Data)

        # --- 验证 2：价格 Forward Fill (InstB) ---
        # InstB (index 1) 在 Day 2 (index 2) 是缺失的
        # Day 1: 201.0 -> Day 2: 应该也是 201.0
        self.assertEqual(price_view[1, 1], 201.0) # T1
        self.assertEqual(price_view[1, 2], 201.0) # T2 (Filled)
        self.assertEqual(price_view[1, 3], 203.0) # T3 (Real Data)

        self.assertEqual(price_view[2, 0], 300.0) # T0 (Back FIlled)
        self.assertEqual(price_view[2, 1], 300.0) # T1 
        self.assertEqual(price_view[2, 2], 301.0) # T2 (Real Data)

        # --- 验证 3：成交量 Zero Fill (InstA) ---
        # InstA 在 Day 1 是缺失的
        # 应该填充为 0
        self.assertEqual(vol_view[0, 0], 1000.0) # T0
        self.assertEqual(vol_view[0, 1], 0.0)    # T1 (Filled 0)
        self.assertEqual(vol_view[0, 2], 1200.0) # T2

    def test_indexed_to_dense_conversion(self):
        """
        测试从 Indexed Block 到 Dense Block 的转换，
        重点验证 'price' (ffill) 和 'volume' (fill 0) 的行为。
        """
        # fill_methods: 1 for price, 0 for volume
        dense_block = PanelBlockDense.from_indexed_block(
            self.block_indexed,
            required_columns=['price', 'volume'],
            fill_methods=[2, 2], 
            frequency='1D',
            inst_cats=self.all_instruments
        )
        price_view = dense_block.get_view('price')   # Shape: (2, 4) -> (Inst, Time)
        v = price_view[1, 2]
        self.assertTrue(np.isnan(v))

    def test_instrument_alignment(self):
        """测试当 Instruments 列表不匹配或需要对齐时的行为"""
        
        # 只请求 InstA
        subset_insts = np.array(['InstA'])
             
        dense_block = PanelBlockDense.from_indexed_block(
            self.block_indexed,
            required_columns=['price'],
            fill_methods=[1],
            inst_cats=subset_insts
        )
        
        self.assertEqual(dense_block.data.shape, (1, 1, 5))
        self.assertEqual(dense_block.instruments[0], 'InstA')

    def test_dense_to_indexed_conversion(self):
        dense_block = PanelBlockDense.from_indexed_block(
            self.block_indexed,
            required_columns=['price', 'volume'],
            fill_methods=[1],
            inst_cats=self.all_instruments
        )
        indexed_block = PanelBlockIndexed.from_dense_block(dense_block)
        self.assertEqual(indexed_block.data.shape, (15, 2))
        self.assertListEqual(list(indexed_block.data.index.names), ['datetime', 'instrument'])
        self.assertListEqual(list(indexed_block.data.columns), ['price', 'volume'])
        self.assertEqual(indexed_block.data.loc[('2023-01-01', 'InstA')]['price'], 100.0)
        self.assertEqual(indexed_block.data.loc[('2023-01-01', 'InstA')]['volume'], 1000)
        self.assertEqual(indexed_block.data.loc[('2023-01-05', 'InstA')]['price'], 104.0)
        self.assertEqual(indexed_block.data.loc[('2023-01-05', 'InstA')]['volume'], 1400)
        self.assertEqual(indexed_block.data.loc[('2023-01-01', 'InstB')]['price'], 200.0)
        self.assertEqual(indexed_block.data.loc[('2023-01-01', 'InstB')]['volume'], 2000)
        self.assertEqual(indexed_block.data.loc[('2023-01-05', 'InstB')]['price'], 204.0)
        self.assertEqual(indexed_block.data.loc[('2023-01-05', 'InstB')]['volume'], 2400)
        self.assertTrue(np.isnan(indexed_block.data.loc[('2023-01-01', 'InstC')]['price']))
        self.assertTrue(np.isnan(indexed_block.data.loc[('2023-01-01', 'InstC')]['volume']))
        self.assertEqual(indexed_block.data.loc[('2023-01-05', 'InstC')]['price'], 304.0)
        self.assertEqual(indexed_block.data.loc[('2023-01-05', 'InstC')]['volume'], 3400)

if __name__ == '__main__':
    unittest.main()