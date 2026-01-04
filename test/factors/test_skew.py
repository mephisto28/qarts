import unittest

import numpy as np
from scipy.stats import skew
from qarts.modeling.factors import create_mock_context,  DailySkewness, ContextSrc, ContextOps


class TestHighOrder(unittest.TestCase):

    def setUp(self):
        self.context = create_mock_context()
        self.ops = ContextOps(self.context)
        self.intraday_return = self.context.get_field(ContextSrc.FACTOR_CACHE, 'intraday_mom')
        self.daily_return = self.context.get_field(ContextSrc.DAILY_QUOTATION, 'daily_return')

    def naive_skewness(self, window: int):
        out = np.empty_like(self.intraday_return)
        daily_return = self.daily_return[:, -window+1:]
        for i in range(self.intraday_return.shape[1]):
            daily_return_ = np.concatenate([daily_return, self.intraday_return[:, i:i+1]], axis=1)
            out[:, i] = skew(daily_return_, axis=1)
        return out

    def assert_arr_equal(self, arr1, arr2, eps=1e-3):
        eps = np.abs(arr1 - arr2)
        eps = eps.mean() / np.abs(arr2).mean()
        self.assertAlmostEqual(eps, 0, places=3, msg=f'eps={eps} arr1={arr1[:5, :5]} arr2={arr2[:5, :5]}')

    def test_skewness(self):
        for window in [5, 50]:
            out1 = np.empty_like(self.context.get_field(ContextSrc.INTRADAY_QUOTATION, 'mid_price'))
            factor = DailySkewness(
                input_fields={
                    ContextSrc.DAILY_QUOTATION: ['daily_return'], 
                    ContextSrc.FACTOR_CACHE: ['intraday_mom']
                }, window=window)
            factor.compute_from_context(self.ops, out1)

            out2 = self.naive_skewness(window)
            self.assert_arr_equal(out1, out2)



if __name__ == '__main__':
    unittest.main()