from typing import Any

import xarray as xr
from matplotlib.figure import Figure

from earthdaily.datacube.constants import (
    DEFAULT_COL_WRAP,
    DEFAULT_COLORMAP,
    DEFAULT_RGB_BLUE,
    DEFAULT_RGB_GREEN,
    DEFAULT_RGB_RED,
    DEFAULT_TIME_INDEX,
    DIM_BANDS,
    DIM_TIME,
)
from earthdaily.datacube.exceptions import DatacubeVisualizationError


def _max_time_wrap(dataset: xr.Dataset, wish: int = DEFAULT_COL_WRAP, col: str = DIM_TIME) -> int:
    if col in dataset.dims:
        return min(wish, dataset.sizes[col])
    return wish


def plot_rgb(
    dataset: xr.Dataset,
    red: str = DEFAULT_RGB_RED,
    green: str = DEFAULT_RGB_GREEN,
    blue: str = DEFAULT_RGB_BLUE,
    col: str = DIM_TIME,
    col_wrap: int = DEFAULT_COL_WRAP,
    background: int | float | None = None,
    **kwargs: Any,
):
    if red not in dataset.data_vars:
        raise DatacubeVisualizationError(f"Band '{red}' not found in dataset")
    if green not in dataset.data_vars:
        raise DatacubeVisualizationError(f"Band '{green}' not found in dataset")
    if blue not in dataset.data_vars:
        raise DatacubeVisualizationError(f"Band '{blue}' not found in dataset")

    ds = dataset
    if isinstance(background, (int, float)):
        ds = ds.copy()
        ds[red] = ds[red].fillna(background)
        ds[green] = ds[green].fillna(background)
        ds[blue] = ds[blue].fillna(background)

    return (
        ds[[red, green, blue]]
        .to_array(dim=DIM_BANDS)
        .plot.imshow(col=col, col_wrap=_max_time_wrap(ds, col_wrap, col=col), **kwargs)
    )


def plot_band(
    dataset: xr.Dataset,
    band: str,
    cmap: str = DEFAULT_COLORMAP,
    col: str = DIM_TIME,
    col_wrap: int = DEFAULT_COL_WRAP,
    **kwargs: Any,
):
    if band not in dataset.data_vars:
        raise DatacubeVisualizationError(f"Band '{band}' not found in dataset")

    return dataset[band].plot.imshow(cmap=cmap, col=col, col_wrap=_max_time_wrap(dataset, col_wrap, col=col), **kwargs)


def plot_mask(
    mask: xr.Dataset | xr.DataArray,
    mask_band: str = "cloud-mask",
    cmap: str = "gray",
    col: str = DIM_TIME,
    col_wrap: int = DEFAULT_COL_WRAP,
    **kwargs: Any,
):
    """
    Plot a cloud-mask (or any boolean/int mask) as an image grid.

    Accepts either a dataset containing the mask band or a standalone DataArray produced
    by masking helpers.
    """
    if isinstance(mask, xr.DataArray):
        data_array = mask
        dataset = mask.to_dataset(name=mask.name or mask_band)
    else:
        if mask_band not in mask.data_vars:
            raise DatacubeVisualizationError(f"Band '{mask_band}' not found in dataset")
        dataset = mask
        data_array = dataset[mask_band]

    return data_array.plot.imshow(
        cmap=cmap,
        col=col,
        col_wrap=_max_time_wrap(dataset, col_wrap, col=col),
        **kwargs,
    )


def thumbnail(
    dataset: xr.Dataset,
    red: str = DEFAULT_RGB_RED,
    green: str = DEFAULT_RGB_GREEN,
    blue: str = DEFAULT_RGB_BLUE,
    time_index: int = DEFAULT_TIME_INDEX,
    **kwargs: Any,
) -> Figure:
    if DIM_TIME in dataset.dims and len(dataset.time) > time_index:
        ds_single = dataset.isel(time=time_index)
    else:
        ds_single = dataset

    if red not in ds_single.data_vars:
        raise DatacubeVisualizationError(f"Band '{red}' not found in dataset")
    if green not in ds_single.data_vars:
        raise DatacubeVisualizationError(f"Band '{green}' not found in dataset")
    if blue not in ds_single.data_vars:
        raise DatacubeVisualizationError(f"Band '{blue}' not found in dataset")

    im = ds_single[[red, green, blue]].to_array(dim=DIM_BANDS).plot.imshow(**kwargs)
    return im.axes.figure
