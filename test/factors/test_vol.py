import unittest

import numpy as np
from qarts.modeling.factors import create_mock_context,  DailyVolatility, ContextSrc, ContextOps


class TestVolatility(unittest.TestCase):

    def setUp(self):
        self.context = create_mock_context()
        self.ops = ContextOps(self.context)
        self.intraday_return = self.context.get_field(ContextSrc.FACTOR_CACHE, 'intraday_mom')
        self.daily_return = self.context.get_field(ContextSrc.DAILY_QUOTATION, 'daily_return')

    def naive_vol(self, window: int):
        out = np.empty_like(self.intraday_return)
        daily_return = self.daily_return[:, -window+1:]
        for i in range(self.intraday_return.shape[1]):
            daily_return_ = np.concatenate([daily_return, self.intraday_return[:, i:i+1]], axis=1)
            out[:, i] = (daily_return_ ** 2).mean(axis=1) ** 0.5
        return out

    def assert_arr_equal(self, arr1, arr2, eps=1e-3):
        eps = np.abs(arr1 - arr2)
        eps = eps.mean() / np.abs(arr2).mean()
        self.assertAlmostEqual(eps, 0, places=3, msg=f'eps={eps} arr1={arr1[:5, :5]} arr2={arr2[:5, :5]}')

    def test_vol(self):
        for window in [2, 5, 50]:
            out1 = np.empty_like(self.context.get_field(ContextSrc.INTRADAY_QUOTATION, 'mid_price'))
            factor = DailyVolatility(
                input_fields={
                    ContextSrc.DAILY_QUOTATION: ['daily_return'], 
                    ContextSrc.FACTOR_CACHE: ['intraday_mom']
                }, window=window)
            factor.compute_from_context(self.ops, out1)

            out2 = self.naive_vol(window)
            breakpoint()
            self.assert_arr_equal(out1, out2)



if __name__ == '__main__':
    unittest.main()