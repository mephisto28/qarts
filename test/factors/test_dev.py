import unittest

import numpy as np
from qarts.modeling.factors import create_mock_context,  MAPriceDeviation, VWAPPriceDeviation, ContextSrc, ContextOps


class TestDeviation(unittest.TestCase):

    def setUp(self):
        self.context = create_mock_context()
        self.ops = ContextOps(self.context)
        self.intraday_price = self.context.get_field(ContextSrc.INTRADAY_QUOTATION, 'mid_price')
        self.daily_price = self.context.get_field(ContextSrc.DAILY_QUOTATION, 'adjusted_close')
        self.daily_volume = self.context.get_field(ContextSrc.DAILY_QUOTATION, 'volume')

    def naive_vwap_deviation(self, window: int):
        vwap = np.nansum(self.daily_price[:, -window:].astype(np.float64) * self.daily_volume[:, -window:], axis=1) / np.nansum(self.daily_volume[:, -window:], axis=1)
        out = (np.log(self.intraday_price) - np.log(vwap)[:, np.newaxis]) / np.sqrt(window)
        return out

    def naive_ma_deviation(self, window: int):
        ma = np.nanmean(self.daily_price[:, -window:], axis=1)
        out = (np.log(self.intraday_price) - np.log(ma)[:, np.newaxis]) / np.sqrt(window)
        return out

    def assert_arr_equal(self, arr1, arr2, eps=1e-3):
        eps = np.abs(arr1 - arr2)
        eps = eps.mean() / np.abs(arr2).mean()
        self.assertAlmostEqual(eps, 0, places=3, msg=f'eps={eps} arr1={arr1[:5, :5]} arr2={arr2[:5, :5]}')

    def test_ma_deviation(self):
        for window in [2, 5, 50]:
            out1 = np.empty_like(self.context.get_field(ContextSrc.INTRADAY_QUOTATION, 'mid_price'))
            factor = MAPriceDeviation(
                input_fields={ContextSrc.DAILY_QUOTATION: ['adjusted_close'], 
                ContextSrc.INTRADAY_QUOTATION: ['mid_price']
            }, window=window)
            factor.compute_from_context(self.ops, out1)

            out2 = self.naive_ma_deviation(window)
            self.assert_arr_equal(out1, out2)

    def test_vwap_deviation(self):
        for window in [2, 5, 50]:
            out = np.empty_like(self.context.get_field(ContextSrc.INTRADAY_QUOTATION, 'mid_price'))
            factor = VWAPPriceDeviation(
                input_fields={ContextSrc.DAILY_QUOTATION: ['adjusted_close', 'volume'], 
                ContextSrc.INTRADAY_QUOTATION: ['mid_price']
            }, window=window)
            factor.compute_from_context(self.ops, out)

            out2 = self.naive_vwap_deviation(window)
            self.assert_arr_equal(out, out2)


if __name__ == '__main__':
    unittest.main()