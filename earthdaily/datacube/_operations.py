from typing import Any

import geopandas as gpd
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

from earthdaily.datacube._geometry import geometry_to_geopandas
from earthdaily.datacube.constants import DIM_TIME
from earthdaily.datacube.exceptions import DatacubeMergeError, DatacubeOperationError
from earthdaily.datacube.models import CompatType


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


def select_time_range(dataset: xr.Dataset, start: str | None = None, end: str | None = None) -> xr.Dataset:
    if DIM_TIME not in dataset.dims:
        raise DatacubeOperationError("Dataset does not have a time dimension")

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
