import unittest

import numpy as np
from scipy.stats import kurtosis
from qarts.modeling.factors import create_mock_context,  DailyKurtosis, ContextSrc, ContextOps

class TestKurtosis(unittest.TestCase):

    def setUp(self):
        self.context = create_mock_context(10, seed=42)
        self.ops = ContextOps(self.context)
        self.daily_return = self.context.get_field(ContextSrc.DAILY_QUOTATION, 'daily_return')
        self.intraday_return = self.context.get_field(ContextSrc.FACTOR_CACHE, 'intraday_mom')

    def naive_kurtosis(self, window: int):
        out = np.empty_like(self.intraday_return)
        daily_return = self.daily_return[:, -window+1:]
        for i in range(self.intraday_return.shape[1]):
            daily_return_ = np.concatenate([daily_return, self.intraday_return[:, i:i+1]], axis=1)
            out[:, i] = kurtosis(daily_return_, axis=1, fisher=True, bias=False)
        return out

    def assert_arr_equal(self, arr1, arr2, eps=1e-3):
        eps = np.abs(arr1 - arr2)
        eps = eps.mean() / np.abs(arr2).mean()
        self.assertAlmostEqual(eps, 0, places=3, msg=f'eps={eps} arr1={arr1[:5, :5]} arr2={arr2[:5, :5]}')

    def test_kurtosis_calculation(self):
        # Test logic
        for window in [10, 20, 30]:
            # print(f"Testing Window={window}...")
            factor = DailyKurtosis(
                input_fields={
                    ContextSrc.DAILY_QUOTATION: ['daily_return'], 
                    ContextSrc.FACTOR_CACHE: ['intraday_mom']
                }, window=window
            )
            
            out_fast = np.zeros_like(self.intraday_return)
            factor.compute_from_context(self.ops, out_fast)
            
            out_naive = self.naive_kurtosis(window)
            
            self.assert_arr_equal(out_fast, out_naive)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)