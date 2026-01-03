import numba as nb
import numpy as np


@nb.njit(parallel=True)
def ffill2d(a, out, reverse: bool = False):
    """
    Out-of-place forward fill along axis=1 for 2D float array.
    out must be preallocated with same shape as a.
    """
    n0, n1 = a.shape
    for i in nb.prange(n0):
        last = np.nan
        if reverse:
            j_range = range(n1 - 1, -1, -1)
        else:
            j_range = range(n1)
        for j in j_range:
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
    

@nb.njit(parallel=True)
def reverse_cumsum_2d(a: np.ndarray, out: np.ndarray):
    n0, n1 = a.shape
    for i in nb.prange(n0):
        acc = 0
        for j in range(n1 - 1, -1, -1):
            v = a[i, j]
            if v == v:
                acc += v
            out[i, j] = acc
    return out