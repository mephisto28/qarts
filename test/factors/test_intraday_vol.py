import unittest
import numpy as np

from qarts.modeling.factors import create_mock_context, ContextOps, ContextSrc, TodayVolatility


class TestTodayVolatility(unittest.TestCase):

    def setUp(self):
        self.context = create_mock_context()
        self.ops = ContextOps(self.context)
        self.mid_price = self.context.get_field(ContextSrc.INTRADAY_QUOTATION, "mid_price")

    @staticmethod
    def _naive_today_volatility_from_midprice(mid_price: np.ndarray) -> np.ndarray:
        n, t = mid_price.shape
        out = np.full((n, t), 0, dtype=np.float64)

        ret = np.full((n, t), 0, dtype=np.float64)
        denom = mid_price[:, :-1]
        numer = mid_price[:, 1:]
        valid = np.isfinite(denom) & (denom != 0.0) & np.isfinite(numer)
        r = np.full_like(numer, np.nan, dtype=np.float64)
        r[valid] = np.log(numer[valid] / denom[valid])
        ret[:, 1:] = r

        for ti in range(t):
            if ti < 1:
                continue
            x = ret[:, 1:ti + 1]
            m = np.isfinite(x)
            ss = np.nansum(x * x, axis=1)
            cnt = m.sum(axis=1).astype(np.float64)
            ok = cnt > 0
            out[ok, ti] = np.sqrt(ss[ok] / cnt[ok])
        return out

    def assert_arr_equal(self, arr1, arr2, eps=1e-6):
        m = np.isfinite(arr1) & np.isfinite(arr2)
        if not np.any(m):
            return
        denom = np.mean(np.abs(arr2[m])) + 1e-12
        rel = np.mean(np.abs(arr1[m] - arr2[m])) / denom
        self.assertAlmostEqual(rel, 0.0, places=6, msg=f"rel={rel} arr1={arr1[:3,:8]} arr2={arr2[:3,:8]}")

    def test_today_volatility(self):
        out1 = np.empty_like(self.mid_price, dtype=np.float64)
        factor = TodayVolatility(input_fields={ContextSrc.FACTOR_CACHE: ["intraday_mom"]})
        factor.compute_from_context(self.ops, out1)

        out2 = self._naive_today_volatility_from_midprice(self.mid_price)
        self.assert_arr_equal(out1, out2)


if __name__ == "__main__":
    unittest.main()
