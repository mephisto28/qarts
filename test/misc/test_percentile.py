import unittest
import numpy as np
from scipy.stats import rankdata, pearsonr
from qarts.modeling.factors.kernels import fast_binned_percentile_2d


class TestBinnedPercentile(unittest.TestCase):
    
    def setUp(self):
        # 构造测试数据：50行 (Time), 4000列 (Stocks)
        # 使用 t 分布产生一些离群值
        np.random.seed(42)
        self.n_rows = 50
        self.n_cols = 4000
        self.data = np.random.standard_t(df=3, size=(self.n_rows, self.n_cols))

    def test_sample(self):
        a = np.array([np.arange(50), np.arange(50, 0, -1)]).astype(np.float32)
        a[0, 0] = np.nan
        a[1, 1] = np.nan
        result = fast_binned_percentile_2d(a, n_bins=1000, sigma_clip=3.5)
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertTrue(np.isnan(result[1, 1]))
        self.assertTrue(~np.isnan(result[0, 1]))
        
    def test_accuracy_vs_scipy(self):
        """测试近似 Rank 与 Scipy 真实 Rank 的一致性"""
        n_bins = 4000
        # 调用 Numba 函数
        result = fast_binned_percentile_2d(self.data, n_bins=n_bins, sigma_clip=4.0)
        
        for i in range(self.n_rows):
            row = self.data[i]
            true_rank = (rankdata(row, method='average') - 0.5) / self.n_cols
            approx_rank = result[i]
            corr, _ = pearsonr(true_rank, approx_rank)
            self.assertGreater(corr, 0.999, f"Correlation too low at row {i}")
            
            mask = (row > np.mean(row) - 3*np.std(row)) & (row < np.mean(row) + 3*np.std(row))
            mae = np.mean(np.abs(true_rank[mask] - approx_rank[mask]))
            self.assertLess(mae, 0.001, f"MAE too high at row {i}: {mae}")


if __name__ == '__main__':
    unittest.main()