"""
Created on Fri Jun  7 16:19:54 2024

@author: nkk
"""

import xarray as xr
import numpy as np
from scipy.linalg import solve_banded
import warnings


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

    resampled = dataset.resample(time='1D').interpolate('linear')
    weights_binary = np.in1d(resampled.time.dt.date,dataset.time.dt.date)
    if weights is not None:
        weights = np.where(weights_binary==1,weights,weights_binary)
    else:
        weights = weights_binary

    _core_dims = [dim for dim in dataset.dims if dim != "time"]
    _core_dims.extend([time])
    dataset_w = xr.apply_ufunc(
        _whitw,
        resampled,
        input_core_dims=[_core_dims],
        output_core_dims=[_core_dims],
        dask="forbidden",
        vectorize=True,
        kwargs=dict(beta=beta, weights=weights.copy()))
    
    return dataset_w.isel(time=weights_binary)


def _whitw(signal, beta, weights=None):
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
    m = signal.shape[-1]

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

    if weights is None:
        weights = np.ones((m,))
    ab_mat[3, :] += weights

    # pxx = []
    signal_w = np.empty_like(signal)
    for pixel in np.ndindex(signal.shape[:-1]):
        signal_w[*pixel, :] = _whitw_pixel(signal[*pixel, ...], weights, alpha, ab_mat)
    return signal_w


def _whitw_pixel(signal, weights, alpha, ab_mat):
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
    # print(signal)

    is_nan = np.isnan(signal)

    if np.all(is_nan):
        return signal

    # manage local nans
    weights[is_nan] = 0
    signal = np.where(is_nan, 0, signal)
    return solve_banded((alpha, alpha), ab_mat, weights * signal).astype(np.float64)
