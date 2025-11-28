import numpy as np
import xarray as xr

from earthdaily.datacube._whittaker import whittaker_smooth as _whittaker_smooth_impl
from earthdaily.datacube.constants import DEFAULT_TEMPORAL_FREQ, DEFAULT_WHITTAKER_BETA, DIM_TIME
from earthdaily.datacube.exceptions import DatacubeOperationError
from earthdaily.datacube.models import AggregationMethod


def whittaker_smooth(
    dataset: xr.Dataset,
    beta: float = DEFAULT_WHITTAKER_BETA,
    weights: np.ndarray | list | None = None,
    time: str = DIM_TIME,
) -> xr.Dataset:
    result_dataset = dataset.copy()
    for var in dataset.data_vars:
        result_dataset[var] = _whittaker_smooth_impl(dataset[[var]], beta=beta, weights=weights, time=time)[var]
    return result_dataset


def temporal_aggregate(dataset: xr.Dataset, method: AggregationMethod, freq: str = DEFAULT_TEMPORAL_FREQ) -> xr.Dataset:
    if DIM_TIME not in dataset.dims:
        raise DatacubeOperationError("Dataset does not have a time dimension")

    resampled = dataset.resample(time=freq)

    if method == "mean":
        return resampled.mean()
    elif method == "median":
        return resampled.median()
    elif method == "min":
        return resampled.min()
    elif method == "max":
        return resampled.max()
    elif method == "sum":
        return resampled.sum()
    elif method == "std":
        return resampled.std()
    elif method == "var":
        return resampled.var()
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")
