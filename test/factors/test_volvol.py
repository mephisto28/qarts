import unittest

import numpy as np
from qarts.modeling.factors import create_mock_context, DailyVolVol, ContextSrc, ContextOps


class TestVolVol(unittest.TestCase):

    def setUp(self):
        self.context = create_mock_context()
        self.ops = ContextOps(self.context)
        self.intraday_return = self.context.get_field(ContextSrc.FACTOR_CACHE, 'intraday_mom')
        self.daily_return = self.context.get_field(ContextSrc.DAILY_QUOTATION, 'daily_return')

    def naive_volvol(self, vol_window: int, volvol_window: int) -> np.ndarray:
        import pandas as pd
        out = np.empty_like(self.intraday_return)
        for t in range(self.intraday_return.shape[1]):
            daily_return = np.concatenate([self.daily_return, self.intraday_return[:, t:t+1]], axis=1)
            ts = pd.DataFrame(daily_return.T)
            vol = (ts ** 2).rolling(window=vol_window).mean() ** 0.5
            volvol = vol.rolling(window=volvol_window).std(ddof=0)
            out[:, t] = volvol.iloc[-1]
        return out


    def assert_arr_equal(self, arr1, arr2, eps=1e-3):
        eps = np.abs(arr1 - arr2)
        eps = eps.mean() / np.abs(arr2).mean()
        self.assertAlmostEqual(eps, 0, places=3, msg=f'eps={eps} arr1={arr1[:5, :5]} arr2={arr2[:5, :5]}')

    def test_volvol(self):
        vacancy_index = 1
        for window in [2, 5, 30]:
            # out = self.context.get_field(ContextSrc.FACTOR_CACHE, 'vacancy_0')
            # factor = DailyVolatility(
            #     input_fields={
            #         ContextSrc.DAILY_QUOTATION: ['daily_return'], 
            #         ContextSrc.FACTOR_CACHE: ['intraday_mom']
            #     }, window=window)
            # factor.compute_from_context(self.ops, out)
            # self.context.blocks[ContextSrc.FACTOR_CACHE].fields[vacancy_index] = 'daily_volatility_'+str(window)

            out1 = np.empty_like(self.context.get_field(ContextSrc.INTRADAY_QUOTATION, 'mid_price'))
            factor = DailyVolVol(
                input_fields={
                    ContextSrc.DAILY_QUOTATION: ['daily_return'], 
                    ContextSrc.FACTOR_CACHE: ['intraday_mom']
                }, window=window, window2=window*2)
            factor.compute_from_context(self.ops, out1)

            out2 = self.naive_volvol(window, window * 2)
            self.assert_arr_equal(out1, out2)



if __name__ == '__main__':
    unittest.main()