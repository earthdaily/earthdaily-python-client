"""
Created on Fri Jun  7 16:19:54 2024

@author: nkk
"""

import xarray as xr
import numpy as np
from scipy.linalg import solve_banded
import warnings
from dask import array as da


def whittaker(dataset, beta=10000.0, weights=None, time="time"):
    """


    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    beta : TYPE, optional
        DESCRIPTION. The default is 10000.0.
    weights : TYPE, optional
        DESCRIPTION. The default is None.
    time : TYPE, optional
        DESCRIPTION. The default is "time".

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    resampled = dataset.resample({time: "1D"}).interpolate("linear")
    weights_binary = np.in1d(resampled[time].dt.date, dataset[time].dt.date)
    if weights is not None:
        weights_advanced = np.copy(weights_binary.astype(float))
        weights_advanced[weights_advanced == 1.0] = weights
    else:
        weights_advanced = weights_binary

    # _core_dims = [dim for dim in dataset.dims if dim != "time"]
    # _core_dims.extend([time])
    m = resampled[time].size
    ab_mat = _ab_mat(m, beta)

    dataset_w = xr.apply_ufunc(
        _whitw_pixel,
        resampled,
        input_core_dims=[[time]],
        output_core_dims=[[time]],
        output_dtypes=[float],
        dask="parallelized",
        vectorize=True,
        kwargs=dict(weights=weights_advanced, ab_mat=ab_mat),
    )

    return dataset_w.isel({time: weights_binary})


def _ab_mat(m, beta):
    """
    Implement weighted whittaker, only for alpha=3 for efficiency.

    :param signal: 1D signal to smooth
    :type signal: numpy array or list
    :param weights: weights of each sample (one by default)
    :type weights: numpy array or list
    :param alpha: derivation order
    :type alpha: int
    :param beta: penalization parameter
    :type beta: float
    :return: a smooth signal
    :rtype: numpy array
    """
    alpha = 3

    ab_mat = np.zeros((2 * alpha + 1, m))

    ab_mat[0, 3:] = -1.0

    ab_mat[1, [2, -1]] = 3.0
    ab_mat[1, 3:-1] = 6.0

    ab_mat[2, [1, -1]] = -3.0
    ab_mat[2, [2, -2]] = -12.0
    ab_mat[2, 3:-2] = -15.0

    ab_mat[3, [0, -1]] = 1.0
    ab_mat[3, [1, -2]] = 10.0
    ab_mat[3, [2, -3]] = 19.0
    ab_mat[3, 3:-3] = 20.0

    ab_mat[4, 0:-1] = ab_mat[2, 1:]
    ab_mat[5, 0:-2] = ab_mat[1, 2:]
    ab_mat[6, 0:-3] = ab_mat[0, 3:]

    ab_mat *= beta
    return ab_mat


def _whitw_pixel(signal, weights, ab_mat):
    """
    Implement weighted whittaker, only for alpha=3 for efficiency.

    :param signal: 1D signal to smooth
    :type signal: numpy array or list
    :param weights: weights of each sample (one by default)
    :type weights: numpy array or list
    :param alpha: derivation order
    :type alpha: int
    :param beta: penalization parameter
    :type beta: float
    :return: a smooth signal
    :rtype: numpy array
    """
    ab_mat_ = ab_mat.copy()
    is_nan = np.isnan(signal)

    if np.all(is_nan):
        return signal
    # manage local nans
    weights = np.where(is_nan, 0, weights)
    ab_mat_[3, :] += weights

    signal = np.where(is_nan, 0, signal)
    return solve_banded((3, 3), ab_mat_, weights * signal)
