from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

from earthdaily.datacube._geometry import geometry_to_geopandas
from earthdaily.datacube.constants import DIM_TIME
from earthdaily.datacube.exceptions import DatacubeMergeError, DatacubeOperationError
from earthdaily.datacube.models import CompatType, TimeLike, TimeSelection


def clip_datacube(dataset: xr.Dataset, geometry: str | dict[str, Any] | gpd.GeoDataFrame) -> xr.Dataset:
    gdf = geometry_to_geopandas(geometry).to_crs(dataset.rio.crs)

    try:
        clipped = dataset.rio.clip_box(*gdf.total_bounds)
        clipped = clipped.rio.clip(gdf.geometry)
        return clipped
    except NoDataInBounds as e:
        raise DatacubeOperationError(
            f"No data found in the specified geometry bounds. "
            f"This may occur if the geometry doesn't intersect with the datacube extent, "
            f"or if all data in that area has been masked out. Original error: {str(e)}"
        ) from e


def merge_datacubes(dataset1: xr.Dataset, dataset2: xr.Dataset, compat: CompatType = "override") -> xr.Dataset:
    try:
        merged = xr.merge([dataset1, dataset2], compat=compat, join="outer")
        return merged
    except Exception as e:
        raise DatacubeMergeError(f"Failed to merge datacubes: {str(e)}") from e


def _normalize_time_values(time: TimeSelection) -> list[TimeLike]:
    if isinstance(time, str) or not isinstance(time, (list, tuple, set, frozenset)):
        return [time]
    return list(time)


def _build_time_mask(times: pd.DatetimeIndex, values: list[TimeLike]) -> Any:
    mask = np.zeros(len(times), dtype=bool)

    for value in values:
        # Date-only strings ("2026-04-15") select every acquisition on that day,
        # anything carrying a time component must match exactly.
        if isinstance(value, str) and "T" not in value and ":" not in value:
            mask |= times.date == pd.Timestamp(value).date()
        else:
            mask |= times == pd.Timestamp(value)

    return mask


def select_times(dataset: xr.Dataset, time: TimeSelection) -> xr.Dataset:
    sorted_dataset = dataset.sortby(DIM_TIME)
    values = _normalize_time_values(time)

    try:
        mask = _build_time_mask(pd.DatetimeIndex(sorted_dataset[DIM_TIME].values), values)
    except (ValueError, TypeError) as e:
        raise DatacubeOperationError(f"Invalid time value in {values}: {str(e)}") from e

    if not mask.any():
        available_times = sorted_dataset.time.values
        raise DatacubeOperationError(
            f"No data found for time(s) {values}. Available time range: {available_times[0]} to {available_times[-1]}"
        )

    return sorted_dataset.isel({DIM_TIME: np.flatnonzero(mask)})


def select_time_range(
    dataset: xr.Dataset, start: str | None = None, end: str | None = None, time: TimeSelection | None = None
) -> xr.Dataset:
    if DIM_TIME not in dataset.dims:
        raise DatacubeOperationError("Dataset does not have a time dimension")

    if time is not None:
        if start is not None or end is not None:
            raise DatacubeOperationError("Cannot combine 'time' with 'start' or 'end'")
        return select_times(dataset, time)

    if start is None and end is None:
        return dataset

    sorted_dataset = dataset.sortby(DIM_TIME)

    try:
        if start is not None and end is not None:
            result = sorted_dataset.sel(time=slice(start, end))
        elif start is not None:
            result = sorted_dataset.sel(time=slice(start, None))
        else:
            result = sorted_dataset.sel(time=slice(None, end))

        if len(result.time) == 0:
            available_times = sorted_dataset.time.values
            raise DatacubeOperationError(
                f"No data found in time range [{start}, {end}]. "
                f"Available time range: {available_times[0]} to {available_times[-1]}"
            )

        return result
    except KeyError as e:
        available_times = sorted_dataset.time.values
        raise DatacubeOperationError(
            f"Time selection failed: {str(e)}. Available time range: {available_times[0]} to {available_times[-1]}"
        ) from e


def rechunk_datacube(dataset: xr.Dataset, chunks: dict[str, Any]) -> xr.Dataset:
    return dataset.chunk(chunks)
