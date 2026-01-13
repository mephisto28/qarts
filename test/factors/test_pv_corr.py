import unittest

import numpy as np
from qarts.modeling.factors import create_mock_context, ContextOps, ContextSrc, ReturnVolumeCorr


class TestReturnVolumeCorr(unittest.TestCase):

    def setUp(self):
        # Setup basic mock data
        self.context = create_mock_context()
        self.ops = ContextOps(self.context)
        
        # Field Names
        self.f_d_ret = 'daily_return'
        self.f_d_vol = 'adjusted_volume'
        self.f_i_ret = 'intraday_mom'
        self.f_i_vol = 'volume'
        
        # Get Data Refs for verification
        self.d_ret = self.context.get_field(ContextSrc.DAILY_QUOTATION, self.f_d_ret)
        self.d_vol = self.context.get_field(ContextSrc.DAILY_QUOTATION, self.f_d_vol)
        self.i_ret = self.context.get_field(ContextSrc.FACTOR_CACHE, self.f_i_ret)
        self.i_vol = self.context.get_field(ContextSrc.INTRADAY_QUOTATION, self.f_i_vol)

    def naive_corr(self, window: int):
        N, T = self.i_ret.shape
        out = np.full((N, T), np.nan)
        
        # Slice relevant history: last (window - 1) days
        # Note: history array is typically large, take the tail
        hist_len = window - 1
        if hist_len > 0:
            h_r = self.d_ret[:, -hist_len:]
            h_v = self.d_vol[:, -hist_len:]
        else:
            h_r = np.zeros((N, 0))
            h_v = np.zeros((N, 0))

        for t in range(T):
            # Combine history with today's slice (t)
            # Shape (N, Window)
            curr_r = np.concatenate([h_r, self.i_ret[:, t:t+1]], axis=1)
            curr_v = np.concatenate([h_v, self.i_vol[:, t:t+1]], axis=1)
            
            for n in range(N):
                row_r = curr_r[n]
                row_v = curr_v[n]
                
                # Basic NaN filtering
                mask = ~np.isnan(row_r) & ~np.isnan(row_v)
                if mask.sum() < 2:
                    continue
                    
                # Numpy corrcoef returns matrix [[1, r], [r, 1]]
                # Handling standard deviation is zero case implicit in corrcoef (returns nan)
                try:
                    res = np.corrcoef(row_r[mask], row_v[mask])
                    out[n, t] = res[0, 1]
                except Exception:
                    pass
        return out

    def assert_arr_equal(self, arr1, arr2, eps=1e-4):
        # Mask both nan
        arr1 = arr1[:, -1]
        arr2 = arr2[:, -1]
        mask = ~(np.isnan(arr1) & np.isnan(arr2))
        diff = np.abs(arr1[mask] - arr2[mask])
        # Check mean error relative to scale
        # If arrays are all nan, diff is empty
        if diff.size > 0:
            err = diff.mean()
            self.assertLess(err, eps, msg=f'Mean Diff: {err}, \narr1={arr1} \narr2={arr2}')
        
    def test_correlation(self):
        # Test Multiple Windows
        for window in [5, 20]:
            factor = ReturnVolumeCorr(
                input_fields={
                    ContextSrc.DAILY_QUOTATION: [self.f_d_ret, self.f_d_vol], 
                    ContextSrc.FACTOR_CACHE: [self.f_i_ret],
                    ContextSrc.INTRADAY_QUOTATION: [self.f_i_vol]
                }, window=window)
            
            out_fast = np.zeros_like(self.i_ret)
            factor.compute_from_context(self.ops, out_fast)
            
            out_naive = self.naive_corr(window)
            
            self.assert_arr_equal(out_fast, out_naive)


if __name__ == '__main__':
    unittest.main()