import xarray as xr
import numpy as np
from ._pywapor_core import _wt1, _wt2, cve1, second_order_diff_matrix, dist_to_finite
import logging as log


def xr_dist_to_finite(y, dim="time"):
    if dim not in y.dims:
        raise ValueError

    out = xr.apply_ufunc(
        dist_to_finite,
        y,
        y[dim],
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=False,
        dask="parallelized",
    )

    return out


def xr_choose_func(y, lmbd, dim):
    funcs = [_wt1, _wt2]
    y_dims = getattr(y, "ndim", 0)
    lmbd_dims = getattr(lmbd, "ndim", 0)
    if y_dims in [2, 3] and lmbd_dims in [1]:
        wt_func = funcs[1]
        icd = [[dim], [], ["lmbda"], [], [], [], [], []]
        ocd = [["lmbda", dim]]
    elif y_dims in [2] and lmbd_dims in [2]:
        raise ValueError
    else:
        wt_func = funcs[0]
        icd = [[dim], [], [], [], [], [], [], []]
        ocd = [[dim]]

    return wt_func, icd, ocd


def assert_lmbd(lmbd):
    # Check lmbdas.
    if isinstance(lmbd, float) or isinstance(lmbd, int) or isinstance(lmbd, list):
        lmbd = np.array(lmbd)
    assert lmbd.ndim <= 2
    if isinstance(lmbd, np.ndarray) or np.isscalar(lmbd):
        if not np.isscalar(lmbd):
            assert lmbd.ndim <= 1
            if lmbd.ndim == 0:
                lmbd = float(lmbd)
            else:
                lmbd = xr.DataArray(lmbd, dims=["lmbda"], coords={"lmbda": lmbd})
        # else:
        lmbd = xr.DataArray(lmbd)

    return lmbd


def xr_wt(
    datacube,
    lmbd,
    time="time",
    weights=None,
    a=0.5,
    min_value=-np.inf,
    max_value=np.inf,
    max_iter=10,
):
    datacube = datacube.chunk(time=-1)
    datacube_ = datacube.copy()
    lmbd = assert_lmbd(lmbd)

    # Normalize x-coordinates
    x = datacube[time]
    x = (x - x.min()) / (x.max() - x.min()) * x.size

    # Create x-aware delta matrix.
    A = second_order_diff_matrix(x)

    # Make default u weights if necessary.
    if isinstance(weights, type(None)):
        weights = np.ones(x.shape)

    # Choose which vectorized function to use.
    _wt, icd, ocd = xr_choose_func(datacube, lmbd, time)

    # Make sure lmbd is chunked similar to y.
    if not isinstance(datacube.chunk, type(None)):
        lmbd = lmbd.chunk(
            {
                k: v
                for k, v in datacube.unify_chunks().chunksizes.items()
                if k in lmbd.dims
            }
        )

    # Apply whittaker smoothing along axis.
    datacube = xr.apply_ufunc(
        _wt,
        datacube,
        A,
        lmbd,
        weights,
        a,
        min_value,
        max_value,
        max_iter,
        input_core_dims=icd,
        output_core_dims=ocd,
        dask="allowed",
    )
    return xr.where(np.isnan(datacube_).all(dim=time), datacube_, datacube)
