import unittest

import numpy as np
from scipy.stats import skew
from qarts.modeling.factors import create_mock_context,  TodaySkewness, ContextSrc, ContextOps


class TestTodaySkewness(unittest.TestCase):
    def setUp(self):
        self.context = create_mock_context()
        self.ops = ContextOps(self.context)
        self.mid_price = self.context.get_field(ContextSrc.INTRADAY_QUOTATION, "mid_price")

    def naive_today_skewness(self) -> np.ndarray:
        n, t = self.mid_price.shape
        out = np.full((n, t), np.nan, dtype=np.float64)
        ret = np.full((n, t), np.nan, dtype=np.float64)
        ret[:, 1:] = np.log(self.mid_price[:, 1:] / self.mid_price[:, :-1])
        for ti in range(t):
            if ti < 1:
                continue
            for i in range(n):
                out[i, ti] = skew(ret[i, 1 : ti + 1], bias=False)
        return out

    def assert_arr_equal(self, arr1, arr2, eps=1e-6):
        m = np.isfinite(arr1) & np.isfinite(arr2)
        if not np.any(m):
            return
        denom = np.mean(np.abs(arr2[m])) + 1e-12
        rel = np.mean(np.abs(arr1[m] - arr2[m])) / denom
        self.assertAlmostEqual(rel, 0.0, places=2, msg=f"rel={rel} \narr1={arr1[:3,:8]} \narr2={arr2[:3,:8]}")

    def test_today_skewness(self):
        out1 = np.empty_like(self.mid_price, dtype=np.float64)
        factor = TodaySkewness(input_fields={ContextSrc.FACTOR_CACHE: ["intraday_mom"]})
        factor.compute_from_context(self.ops, out1)

        out2 = self.naive_today_skewness()
        self.assert_arr_equal(out1[:, 100:], out2[:, 100:])


if __name__ == "__main__":
    unittest.main()