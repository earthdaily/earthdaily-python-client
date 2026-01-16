from __future__ import annotations

from typing import Any, Callable

import geopandas as gpd
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from earthdaily.datacube._indices import add_indices
from earthdaily.datacube._masking import apply_cloud_mask
from earthdaily.datacube._operations import (
    clip_datacube,
    merge_datacubes,
    rechunk_datacube,
    select_time_range,
)
from earthdaily.datacube._temporal import temporal_aggregate, whittaker_smooth
from earthdaily.datacube._visualization import plot_band, plot_rgb, thumbnail
from earthdaily.datacube._zonal import compute_zonal_stats
from earthdaily.datacube.constants import (
    DEFAULT_AGGREGATION,
    DEFAULT_COL_WRAP,
    DEFAULT_COLORMAP,
    DEFAULT_RGB_BLUE,
    DEFAULT_RGB_GREEN,
    DEFAULT_RGB_RED,
    DEFAULT_TEMPORAL_FREQ,
    DEFAULT_TIME_INDEX,
    DEFAULT_WHITTAKER_BETA,
    DEFAULT_ZONAL_REDUCERS,
    DIM_TIME,
)
from earthdaily.datacube.models import AggregationMethod, CompatType, GroupByOption


class Datacube:
    def __init__(self, dataset: xr.Dataset, metadata: dict[str, Any] | None = None):
        self._dataset = dataset
        self._metadata = metadata or {}

    @property
    def data(self) -> xr.Dataset:
        return self._dataset

    def _create_new(self, dataset: xr.Dataset, **metadata_updates) -> Datacube:
        new_metadata = {**self._metadata, **metadata_updates}
        return Datacube(dataset, new_metadata)

    def with_dataset(self, dataset: xr.Dataset, **metadata_updates) -> Datacube:
        return self._create_new(dataset, **metadata_updates)

    def apply_mask(
        self,
        mask_band: str,
        mask_dataset: xr.Dataset | None = None,
        custom_mask_function: Callable[[xr.Dataset], xr.DataArray] | None = None,
        include_mask_band: bool = False,
        mask_statistics: bool = True,
        intersects: str | dict[str, Any] | gpd.GeoDataFrame | None = None,
        bbox: list | tuple | None = None,
        fill_value=np.nan,
        round_time: bool = False,
        clear_cover: float | None = None,
        clear_values: list[int] | None = None,
        exclude_values: list[int] | None = None,
    ) -> Datacube:
        masked_dataset = apply_cloud_mask(
            dataset=self._dataset,
            mask_dataset=mask_dataset,
            mask_band=mask_band,
            custom_mask_function=custom_mask_function,
            include_mask_band=include_mask_band,
            mask_statistics=mask_statistics,
            intersects=intersects,
            bbox=bbox,
            fill_value=fill_value,
            round_time=round_time,
            clear_cover=clear_cover,
            clear_values=clear_values,
            exclude_values=exclude_values,
        )
        return self._create_new(masked_dataset)

    def clip(self, geometry: str | dict[str, Any] | gpd.GeoDataFrame) -> Datacube:
        clipped_dataset = clip_datacube(self._dataset, geometry)
        return self._create_new(clipped_dataset)

    def merge(self, other: Datacube, compat: CompatType = "override") -> Datacube:
        if not isinstance(other, Datacube):
            raise TypeError("Can only merge with another Datacube instance")

        merged_dataset = merge_datacubes(self._dataset, other._dataset, compat=compat)
        return self._create_new(merged_dataset)

    def select_time(self, start: str | None = None, end: str | None = None) -> Datacube:
        filtered_dataset = select_time_range(self._dataset, start, end)
        return self._create_new(filtered_dataset)

    def rechunk(self, chunks: dict[str, Any]) -> Datacube:
        rechunked_dataset = rechunk_datacube(self._dataset, chunks)
        return self._create_new(rechunked_dataset)

    def plot_rgb(
        self,
        red: str = DEFAULT_RGB_RED,
        green: str = DEFAULT_RGB_GREEN,
        blue: str = DEFAULT_RGB_BLUE,
        col: str = DIM_TIME,
        col_wrap: int = DEFAULT_COL_WRAP,
        background: int | float | None = None,
        **kwargs: Any,
    ):
        return plot_rgb(self._dataset, red, green, blue, col, col_wrap, background, **kwargs)

    def plot_band(
        self,
        band: str,
        cmap: str = DEFAULT_COLORMAP,
        col: str = DIM_TIME,
        col_wrap: int = DEFAULT_COL_WRAP,
        **kwargs: Any,
    ):
        return plot_band(self._dataset, band, cmap, col, col_wrap, **kwargs)

    def thumbnail(
        self,
        red: str = DEFAULT_RGB_RED,
        green: str = DEFAULT_RGB_GREEN,
        blue: str = DEFAULT_RGB_BLUE,
        time_index: int = DEFAULT_TIME_INDEX,
        **kwargs: Any,
    ) -> Figure:
        return thumbnail(self._dataset, red, green, blue, time_index, **kwargs)

    def add_indices(self, indices: list[str], **kwargs: Any) -> Datacube:
        dataset_with_indices = add_indices(self._dataset, indices, **kwargs)
        return self._create_new(dataset_with_indices)

    def zonal_stats(
        self,
        geometry: str | dict[str, Any] | gpd.GeoDataFrame,
        reducers: list[str] = DEFAULT_ZONAL_REDUCERS,
        all_touched: bool = True,
        preserve_columns: bool = True,
        lazy_load: bool = True,
    ):
        return compute_zonal_stats(self._dataset, geometry, reducers, all_touched, preserve_columns, lazy_load)

    def whittaker(
        self, beta: float = DEFAULT_WHITTAKER_BETA, weights: np.ndarray | list | None = None, time: str = DIM_TIME
    ) -> Datacube:
        smoothed_dataset = whittaker_smooth(self._dataset, beta, weights, time)
        return self._create_new(smoothed_dataset)

    def temporal_aggregate(
        self,
        method: AggregationMethod = DEFAULT_AGGREGATION,
        freq: str = DEFAULT_TEMPORAL_FREQ,
        groupby: GroupByOption | None = None,
    ) -> Datacube:
        aggregated_dataset = temporal_aggregate(self._dataset, method, freq, groupby)
        return self._create_new(aggregated_dataset)

    def resample(self, freq: str, method: AggregationMethod = DEFAULT_AGGREGATION) -> Datacube:
        resampled_dataset = temporal_aggregate(self._dataset, method, freq)
        return self._create_new(resampled_dataset)

    @property
    def bands(self) -> list[str]:
        return list(self._dataset.data_vars)

    @property
    def timestamps(self) -> list:
        if DIM_TIME in self._dataset.dims and len(self._dataset.time) > 0:
            return self._dataset.time.values.tolist()
        return []

    @property
    def crs(self) -> str | None:
        try:
            return str(self._dataset.rio.crs)
        except Exception:
            return None

    @property
    def resolution(self) -> tuple[float, float] | None:
        try:
            res = self._dataset.rio.resolution()
            return (abs(res[0]), abs(res[1]))
        except Exception:
            return None

    @property
    def shape(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self._dataset.sizes.items()}

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        try:
            bounds = self._dataset.rio.bounds()
            return bounds
        except Exception:
            return None

    def info(self) -> str:
        info_lines = [
            "Datacube Information:",
            f"  Dimensions: {self.shape}",
            f"  Bands: {', '.join(self.bands)}",
            f"  CRS: {self.crs}",
            f"  Resolution: {self.resolution}",
            f"  Extent: {self.extent}",
            f"  Timestamps: {len(self.timestamps)} time steps",
        ]
        if self.timestamps:
            info_lines.append(f"  Time range: {self.timestamps[0]} to {self.timestamps[-1]}")
        return "\n".join(info_lines)

    def __repr__(self) -> str:
        return f"Datacube(bands={len(self.bands)}, shape={self.shape}, crs={self.crs})"

    def __str__(self) -> str:
        return self.info()
