import numpy as np
import xarray as xr
from scipy.linalg import solve_banded

from earthdaily.datacube.constants import (
    DEFAULT_WHITTAKER_BETA,
    DEFAULT_WHITTAKER_FREQ,
    DIM_TIME,
    WHITTAKER_ALPHA,
    WHITTAKER_BAND_SOLVE,
)


def whittaker_smooth(
    dataset: xr.Dataset,
    beta: float = DEFAULT_WHITTAKER_BETA,
    weights: np.ndarray | list | None = None,
    time: str = DIM_TIME,
) -> xr.Dataset:
    """
    Apply Whittaker smoothing to a time series dataset.

    Smooths temporal data using Whittaker-Henderson smoothing (Whittaker, 1923) with
    a third-order difference penalty (α=3). The algorithm: (1) resamples data to daily
    frequency using linear interpolation, (2) constructs the penalty matrix D^T D (where D is
    the third-order difference matrix), (3) solves the linear system (W + λD^T D)z = Wy
    for each pixel using banded matrix storage for efficiency with scipy.linalg.solve_banded,
    and (4) returns smoothed values at original observation times. Implementation follows
    Eilers (2003).

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset with a time dimension to smooth.
    beta : float, optional
        Smoothing parameter (λ). Higher values produce smoother results.
        Default is DEFAULT_WHITTAKER_BETA (10000.0).
    weights : np.ndarray | list | None, optional
        Optional weights for each observation. If None, all observations
        are weighted equally. Default is None.
    time : str, optional
        Name of the time dimension. Default is DIM_TIME ("time").

    Returns
    -------
    xr.Dataset
        Smoothed dataset with the same structure as input, but with smoothed
        values along the time dimension.
    """
    resampled = dataset.resample({time: DEFAULT_WHITTAKER_FREQ}).interpolate("linear")
    weights_binary = np.isin(resampled[time].dt.date, dataset[time].dt.date)

    if weights is not None:
        weights_array = np.asarray(weights, dtype=float)
        if weights_array.ndim == 0:
            weights_array = np.full(int(weights_binary.sum()), float(weights_array))

        expected_length = int(weights_binary.sum())
        if weights_array.size != expected_length:
            raise ValueError(
                f"weights must have length {expected_length} (number of original observations), "
                f"got {weights_array.size}"
            )

        weights_advanced = weights_binary.astype(float)
        weights_advanced[weights_binary] = weights_array
    else:
        weights_advanced = weights_binary.astype(float)

    m = resampled[time].size
    ab_mat = _create_banded_matrix(m, beta)

    smoothed = xr.apply_ufunc(
        _whittaker_pixel,
        resampled,
        input_core_dims=[[time]],
        output_core_dims=[[time]],
        output_dtypes=[float],
        dask="parallelized",
        vectorize=True,
        kwargs={"weights": weights_advanced, "ab_mat": ab_mat},
    )

    return smoothed.isel({time: weights_binary})


def _create_banded_matrix(m: int, beta: float) -> np.ndarray:
    """
    Create the banded matrix representation of λD^T D for third-order differences.

    Constructs the penalty matrix D^T D where D is the third-order finite difference
    operator, stored in LAPACK banded storage format for use with scipy.linalg.solve_banded.
    The coefficients are derived from the mathematical structure of the third-order
    difference operator and its transpose product (Eilers, 2003).

    Parameters
    ----------
    m : int
        Size of the time series (number of time points).
    beta : float
        Smoothing parameter (λ) that scales the penalty matrix.

    Returns
    -------
    np.ndarray
        A (7, m) array representing the symmetric banded matrix in LAPACK banded
        storage format. The matrix has bandwidth (3, 3) meaning 3 sub-diagonals
        and 3 super-diagonals.
    """
    ab_mat = np.zeros((2 * WHITTAKER_ALPHA + 1, m))

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


def _whittaker_pixel(signal: np.ndarray, weights: np.ndarray, ab_mat: np.ndarray) -> np.ndarray:
    """
    Apply Whittaker smoothing to a single pixel's time series.

    Solves the linear system (W + λD^T D)z = Wy for a single time series, where W is
    the diagonal weight matrix and D^T D is the penalty matrix stored in banded format.
    Handles NaN values by setting weights to 0 for NaN observations, setting signal values
    to 0 for NaN observations, and returning the original signal unchanged if all values
    are NaN. The system is solved using scipy.linalg.solve_banded for efficiency.

    Parameters
    ----------
    signal : np.ndarray
        1D array of observed values for a single pixel over time.
    weights : np.ndarray
        1D array of weights for each time point (same length as signal).
    ab_mat : np.ndarray
        Banded matrix representation of λD^T D from _create_banded_matrix.

    Returns
    -------
    np.ndarray
        1D array of smoothed values, same length as signal.
    """
    ab_mat_ = ab_mat.copy()
    is_nan = np.isnan(signal)

    if np.all(is_nan):
        return signal

    weights = np.where(is_nan, 0, weights)
    ab_mat_[3, :] += weights

    signal = np.where(is_nan, 0, signal)
    return solve_banded(WHITTAKER_BAND_SOLVE, ab_mat_, weights * signal)
