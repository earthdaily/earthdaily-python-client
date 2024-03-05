"""
@author: Bert Coerver
"""

import numpy as np
from numba import types, guvectorize, njit, float64, int64


def second_order_diff_matrix(x):
    X = np.append(x, [np.nan, np.nan])
    # Create x-aware delta matrix. When sample-points are at equal distance,
    # this reduces to [[1, -2, 1, ...], ..., [..., 1, -2, 1]].
    diag_n0 = 2 / ((X[1:-1] - X[:-2]) * (X[2:] - X[:-2]))
    diag_n1 = -2 / ((X[2:-1] - X[1:-2]) * (X[1:-2] - X[0:-3]))
    diag_n2 = 2 / ((X[2:-2] - X[1:-3]) * (X[2:-2] - X[0:-4]))
    D = np.diag(diag_n0) + np.diag(diag_n1, k=1) + np.diag(diag_n2, k=2)
    D = D[:-2, :]
    A = np.dot(D.T, D)
    return A


@njit(
    types.Tuple((float64[:], float64, float64))(
        float64[:],
        float64[::1],
        float64,
        float64[:],
        float64[:],
        float64,
        float64[:, ::1],
        float64,
    )
)
def iterator(Z, Y, a_, w, u, lmb, A, a):
    v = np.where(Y > Z, a_, 1 - a_)
    W = np.diag(v * w * u)
    Z_ = np.linalg.solve(W + lmb * A, np.dot(W, Y))
    rmse = np.sqrt(np.mean((Z - Z_) ** 2))
    a_ = a
    return Z_, rmse, a_


@njit(
    types.Tuple((float64[:], float64[::1], float64[:], float64, int64, float64))(
        float64[:]
    )
)
def initiator(Y):
    w = np.isfinite(Y, np.zeros_like(Y))
    Y_ = np.where(w == 0, 0, Y)
    a_ = rmse = 0.5
    j = 0
    Z = np.zeros_like(Y)
    return Z, Y_, w, a_, j, rmse


@njit(
    float64[:](
        float64[::1],
        float64[:, ::1],
        float64,
        float64[:],
        float64,
        float64,
        float64,
        int64,
    )
)
def whittaker(Y, A, lmbda, u, a, min_drange, max_drange, max_iter):
    Z, Y_, w, a_, j, rmse = initiator(Y)
    relevant_idxs_size = n_peaks = 0
    while j == 0 or (
        (not (a == 0.5 or rmse < 0.01) or (relevant_idxs_size > 0)) and (j < max_iter)
    ):
        Z[:], rmse, a_ = iterator(Z, Y_, a_, w, u, lmbda, A, a)
        j += 1

        idxs = np.diff(np.sign(Z[1:] - Z[:-1])).nonzero()[0] + 1
        if (j == 1) or (n_peaks != idxs.size):
            z = Z[idxs]
            is_peak_valley = (z > max_drange) | (z < min_drange)
            is_valley = (z < min_drange)[is_peak_valley]
            peaks_valleys = idxs[is_peak_valley]
            peak_valley_n = np.searchsorted(idxs, peaks_valleys)
            inequal_swap = np.where(is_valley, -1, 1)
            peak_limit = np.where(is_valley, min_drange, max_drange)
            orig_z = Z[peaks_valleys]
            n_peaks = idxs.size
            m = 0

        not_solved = (
            np.take(Z, idxs[peak_valley_n]) * inequal_swap > peak_limit * inequal_swap
        )
        m += 1
        relevant_idxs = peaks_valleys[not_solved]
        Y_[relevant_idxs] = peak_limit[not_solved] - 0.2 * m * (
            orig_z[not_solved] - peak_limit[not_solved]
        )
        w[relevant_idxs] = 1.0
        relevant_idxs_size = relevant_idxs.size
    return Z


@guvectorize(
    [
        (
            float64[::1],
            float64[:, ::1],
            float64,
            float64[:],
            float64,
            float64,
            float64,
            int64,
            float64[:],
        )
    ],
    "(n),(n,n),(),(n),(),(),(),()->(n)",
    nopython=True,
    target="parallel",
)
def _wt1(Y, A, lmbda, u, a, min_drange, max_drange, max_iter, Z):
    Z[:] = whittaker(Y, A, lmbda, u, a, min_drange, max_drange, max_iter)


@guvectorize(
    [
        (
            float64[::1],
            float64[:, ::1],
            float64[:],
            float64[:],
            float64,
            float64,
            float64,
            int64,
            float64[:, :],
        )
    ],
    "(n),(n,n),(k),(n),(),(),(),()->(k,n)",
    nopython=True,
    target="parallel",
)
def _wt2(Y, A, lmbda, u, a, min_drange, max_drange, max_iter, Z):
    for i, lmb in enumerate(lmbda):
        Z[i, :] = whittaker(Y, A, lmb, u, a, min_drange, max_drange, max_iter)


def wt(Y, A, lmbdas, u, a, min_drange, max_drange, max_iter):
    funcs = [_wt1, _wt2]
    y_dims = getattr(Y, "ndim", 0)
    lmbdas_dims = getattr(lmbdas, "ndim", 0)
    if y_dims in [2, 3] and lmbdas_dims in [1]:
        wt_func = funcs[1]
    elif y_dims in [2] and lmbdas_dims in [2]:
        raise ValueError
    else:
        wt_func = funcs[0]
    Z = wt_func(Y, A, lmbdas, u, a, min_drange, max_drange, max_iter)
    return Z


def dist_to_finite(y, x, axis=-1, extrapolate=False):
    def _dtf(y, x, extrapolate=False):
        # NOTE could also use `xarray.core.missing._get_nan_block_lengths instead` (similar, but not exactly the same)
        # NOTE can use numba when this is resolved https://github.com/numba/numba/issues/1269
        part1 = np.expand_dims(np.extract(np.isfinite(y), x), 1)
        part2 = np.expand_dims(x, 0)
        distances = part2 - part1
        dist = np.min(np.abs(distances), axis=0, initial=np.timedelta64(365, "D"))
        if not extrapolate and distances.size != 0:
            if distances.shape[0] > 1:
                mask = np.ptp(np.sign(distances), axis=0).astype(int) > 0
            else:
                mask = np.isfinite(y)
            dist = np.where(mask, dist, np.timedelta64(365, "D"))
        return dist

    out = np.apply_along_axis(_dtf, axis, y, x, extrapolate=extrapolate)
    return out


def cve0(lmbdas, Y, A):
    def _cve(lmbdas, Y, A):
        assert len(Y.shape) == 1
        m = len(Y)
        mask = np.isfinite(Y)
        u = np.ones_like(Y)
        Y_ = np.where(np.eye(m), np.nan, np.repeat(Y[np.newaxis, :], m, axis=0))
        z = wt(Y_[mask, :], A, lmbdas, u, 0.5, -np.inf, np.inf, 10)
        y_hat = np.diagonal(z[:, :, mask], axis1=0, axis2=2)
        cves = np.sqrt(np.mean(((Y[mask] - y_hat) ** 2), axis=1))
        return cves

    f = np.vectorize(_cve, signature="(n),(m),(m,m)->(n)")
    cves = f(lmbdas, Y, A)
    return cves


@guvectorize(
    [(float64[:], float64[:], float64[:, ::1], float64[:], float64[:])],
    "(k),(m),(m,m),(m)->(k)",
    nopython=True,
    target="parallel",
)
def cve1(lmbdas, Y, A, u, cves):
    w = np.isfinite(Y, np.zeros_like(Y))
    W = np.diag(w * u)
    Y_ = np.where(w == 0, 0, Y)
    for i, lmbda in enumerate(lmbdas):
        H = np.linalg.solve(W + lmbda * A, W)  # Eq. 13
        y_hat = np.dot(H, Y_)  # Eq. 10
        hii = np.diag(H)
        cves[i, ...] = np.sqrt(np.nanmean(((Y - y_hat) / (1 - hii)) ** 2))  # Eq. 9 + 11


# if __name__ == "__main__":

#     import xarray as xr
#     from pywapor.general.processing_functions import open_ds

#     # fh = r"/Users/hmcoerver/Local/tests/test_22/SENTINEL2/S2MSI2A_R60m.nc"
#     fh = r"/Users/hmcoerver/Local/tests/test_22/VIIRSL1/VNP02IMG.nc"
#     ds = open_ds(fh)

#     y = ds.isel(y=30, x= 30).bt.values
#     x = ds.time.values

#     # Normalize x-coordinates
#     x_ = (x - x.min()) / (x.max() - x.min()) * x.size

#     # Create x-aware delta matrix.
#     A = second_order_diff_matrix(x_)

#     lmbdas = 100.
#     min_drange = -np.inf
#     max_drange = np.inf
#     # min_drange = -1.
#     # max_drange = 1.
#     max_iter = 10
#     u = np.ones(x.shape)
#     a = 0.5

#     z = wt(y, A, lmbdas, u, a, min_drange, max_drange, max_iter)

#     xdist = dist_to_finite(y, x, axis = -1, extrapolate = False)
