import numba as nb
import numpy as np


@nb.njit(parallel=True)
def ffill2d(a, out):
    """
    Out-of-place forward fill along axis=1 for 2D float array.
    out must be preallocated with same shape as a.
    """
    n0, n1 = a.shape
    for i in nb.prange(n0):
        last = np.nan
        for j in range(n1):
            x = a[i, j]
            if x == x:
                last = x
                out[i, j] = x
            else:
                if last == last:
                    out[i, j] = last
                else:
                    out[i, j] = x   # still NaN
    return out
    