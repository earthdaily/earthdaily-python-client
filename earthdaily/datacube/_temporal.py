import numpy as np
import pandas as pd
import xarray as xr

from earthdaily.datacube._whittaker import whittaker_smooth as _whittaker_smooth_impl
from earthdaily.datacube.constants import DEFAULT_TEMPORAL_FREQ, DEFAULT_WHITTAKER_BETA, DIM_TIME
from earthdaily.datacube.exceptions import DatacubeOperationError
from earthdaily.datacube.models import AggregationMethod, GroupByOption


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


def temporal_aggregate(
    dataset: xr.Dataset,
    method: AggregationMethod,
    freq: str = DEFAULT_TEMPORAL_FREQ,
    groupby: GroupByOption | None = None,
) -> xr.Dataset:
    if DIM_TIME not in dataset.dims:
        raise DatacubeOperationError("Dataset does not have a time dimension")

    if groupby is not None:
        try:
            groupby = GroupByOption(groupby)
        except ValueError:
            raise ValueError(f"Unsupported groupby value: {groupby}. Supported: {[e.value for e in GroupByOption]}")

    grouped: xr.core.groupby.DatasetGroupBy | xr.core.resample.DatasetResample
    if groupby is not None:
        grouped = dataset.groupby(groupby)
    else:
        grouped = dataset.resample(time=freq)

    if method == "mean":
        result = grouped.mean()
    elif method == "median":
        result = grouped.median()
    elif method == "min":
        result = grouped.min()
    elif method == "max":
        result = grouped.max()
    elif method == "sum":
        result = grouped.sum()
    elif method == "std":
        result = grouped.std()
    elif method == "var":
        result = grouped.var()
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")

    if groupby == GroupByOption.DATE:
        result = result.rename({"date": "time"})
        result = result.assign_coords(time=pd.to_datetime(result.time.values))

    return result
