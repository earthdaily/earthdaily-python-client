import warnings
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.features import geometry_mask

from earthdaily.datacube._geometry import bbox_to_geopandas, geometry_to_geopandas
from earthdaily.datacube.constants import DEFAULT_BBOX_CRS, DIM_TIME, DIM_X, DIM_Y, PERCENT_MAX, PERCENT_MIN
from earthdaily.datacube.exceptions import DatacubeMaskingError


def apply_cloud_mask(
    dataset: xr.Dataset,
    mask_band: str = "cloud-mask",
    mask_dataset: xr.Dataset | None = None,
    custom_mask_function: Callable[[xr.Dataset], xr.DataArray] | None = None,
    include_mask_band: bool = False,
    mask_statistics: bool = True,
    intersects: str | dict[str, Any] | gpd.GeoDataFrame | None = None,
    bbox: list | tuple | None = None,
    bbox_crs: str = DEFAULT_BBOX_CRS,
    fill_value=np.nan,
    round_time: bool = False,
    clear_cover: float | None = None,
    clear_values: list[int] | None = None,
    exclude_values: list[int] | None = None,
) -> xr.Dataset:
    """
    Apply cloud masking to a dataset using a mask band.

    Masks dataset variables based on a mask band, replacing masked pixels with fill_value.
    Supports three masking modes: (1) clear_values - keep pixels matching specified values,
    (2) exclude_values - exclude pixels matching specified values, or (3) custom_mask_function
    - use a custom function to generate the mask. Optionally computes mask statistics (clear
    pixel counts and percentages) and filters by minimum clear coverage threshold.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset to mask. Must contain mask_band after merging with mask_dataset
        if provided.
    mask_band : str, optional
        Name of the mask band variable. Default is "cloud-mask".
    mask_dataset : xr.Dataset | None, optional
        Optional separate dataset containing the mask band. If provided, merged with
        dataset before masking. The mask_band can be in either dataset or mask_dataset.
        Default is None.
    custom_mask_function : Callable[[xr.Dataset], xr.DataArray] | None, optional
        Custom function that takes a dataset and returns a boolean DataArray mask.
        Must return True for pixels to keep. Default is None.
    include_mask_band : bool, optional
        Whether to include the mask band in the output dataset. Default is False.
    mask_statistics : bool, optional
        Whether to compute and add clear_pixels and clear_percent coordinates.
        Default is True.
    intersects : str | dict[str, Any] | gpd.GeoDataFrame | None, optional
        Geometry for computing mask statistics within a specific area. Can be WKT string,
        GeoJSON dict, or GeoDataFrame. Default is None.
    bbox : list | tuple | None, optional
        Bounding box [min_x, min_y, max_x, max_y] for computing mask statistics.
        Default is None.
    bbox_crs : str, optional
        CRS for the bbox parameter. Default is DEFAULT_BBOX_CRS ("EPSG:4326").
    fill_value : Any, optional
        Value to use for masked pixels. Default is np.nan.
    round_time : bool, optional
        Whether to round time coordinate to seconds precision. Default is False.
    clear_cover : float | None, optional
        Minimum clear coverage percentage (0-100) threshold. Time steps below this
        threshold are filtered out. Requires mask_statistics=True. Default is None.
    clear_values : list[int] | None, optional
        List of mask band values that represent clear pixels. Pixels with these values
        are kept. Must provide one of clear_values, exclude_values, or custom_mask_function.
        Default is None.
    exclude_values : list[int] | None, optional
        List of mask band values to exclude. Pixels with these values are masked out.
        Must provide one of clear_values, exclude_values, or custom_mask_function.
        Default is None.

    Returns
    -------
    xr.Dataset
        Masked dataset with masked pixels set to fill_value. If mask_statistics=True,
        includes clear_pixels and clear_percent coordinates. If clear_cover is specified,
        filtered to time steps meeting the threshold.

    Raises
    ------
    DatacubeMaskingError
        If mask_band is not found, masking mode is not specified, mask dimensions are
        incompatible, or clear_cover filtering fails.
    """
    result_dataset = dataset.copy()
    if mask_dataset is not None:
        result_dataset = xr.merge([result_dataset, mask_dataset], compat="override")
    _ensure_mask_band(result_dataset, mask_band)
    if round_time and DIM_TIME in result_dataset.dims:
        _round_time_coordinate(result_dataset)
    mask = _build_mask_array(result_dataset, mask_band, custom_mask_function, clear_values, exclude_values)
    _validate_mask_dimensions(result_dataset, mask_band, mask)
    result_dataset = _apply_mask_to_variables(result_dataset, mask_band, mask, fill_value)
    if not include_mask_band and mask_band in result_dataset.data_vars:
        result_dataset = result_dataset.drop_vars(mask_band)
    if mask_statistics:
        result_dataset = _apply_mask_statistics(result_dataset, mask, intersects, bbox, bbox_crs)
    if clear_cover is not None:
        result_dataset = _filter_by_clear_cover(result_dataset, clear_cover)
    return result_dataset


def _ensure_mask_band(dataset: xr.Dataset, mask_band: str) -> None:
    """
    Validate that the mask band exists in the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to check.
    mask_band : str
        Name of the mask band to validate.

    Raises
    ------
    DatacubeMaskingError
        If mask_band is not found in the dataset.
    """
    if mask_band not in dataset:
        raise DatacubeMaskingError(f"'{mask_band}' not found in dataset")


def _round_time_coordinate(dataset: xr.Dataset) -> None:
    """
    Round time coordinate to seconds precision.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with time coordinate to round.

    Raises
    ------
    DatacubeMaskingError
        If time coordinate is not datetime type.
    """
    if not np.issubdtype(dataset[DIM_TIME].dtype, np.datetime64):
        raise DatacubeMaskingError("round_time requires time coordinate to be datetime type")
    dataset[DIM_TIME] = dataset.time.dt.round("s")


def _build_mask_array(
    dataset: xr.Dataset,
    mask_band: str,
    custom_mask_function: Callable[[xr.Dataset], xr.DataArray] | None,
    clear_values: list[int] | None,
    exclude_values: list[int] | None,
) -> xr.DataArray:
    """
    Build boolean mask array from mask band data.

    Constructs a boolean mask using one of three methods: custom function, clear_values,
    or exclude_values. Returns True for pixels to keep, False for pixels to mask.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the mask band.
    mask_band : str
        Name of the mask band variable.
    custom_mask_function : Callable[[xr.Dataset], xr.DataArray] | None
        Custom function to generate mask, or None.
    clear_values : list[int] | None
        Mask values representing clear pixels, or None.
    exclude_values : list[int] | None
        Mask values to exclude, or None.

    Returns
    -------
    xr.DataArray
        Boolean mask array with True for pixels to keep.

    Raises
    ------
    DatacubeMaskingError
        If no masking mode is specified or custom_mask_function returns invalid type.
    """
    cloud_mask_data = dataset[mask_band]
    if custom_mask_function is not None:
        mask = custom_mask_function(dataset)
        if not isinstance(mask, xr.DataArray):
            raise DatacubeMaskingError("custom_mask_function must return an xarray DataArray")
    elif clear_values is not None:
        mask = cloud_mask_data.isin(clear_values)
    elif exclude_values is not None:
        mask = ~cloud_mask_data.isin(exclude_values)
    else:
        raise DatacubeMaskingError("Must provide one of: clear_values, exclude_values, or custom_mask_function")
    return mask


def _validate_mask_dimensions(dataset: xr.Dataset, mask_band: str, mask: xr.DataArray) -> None:
    """
    Validate that mask dimensions are compatible with dataset variables.

    Ensures mask can be applied to all variables by checking that mask dimensions
    are a subset of each variable's dimensions.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to validate against.
    mask_band : str
        Name of the mask band (excluded from validation).
    mask : xr.DataArray
        Mask array to validate.

    Raises
    ------
    DatacubeMaskingError
        If mask has dimensions not present in any dataset variable.
    """
    mask_dims = set(mask.dims)
    for var in dataset.data_vars:
        if var == mask_band:
            continue
        var_dims = set(dataset[var].dims)
        extra_dims = mask_dims - var_dims
        if extra_dims:
            raise DatacubeMaskingError(
                f"Cannot mask variable '{var}' with mask dimensions {mask.dims} "
                f"and variable dimensions {dataset[var].dims}; "
                f"extra dimensions present."
            )


def _apply_mask_to_variables(
    dataset: xr.Dataset,
    mask_band: str,
    mask: xr.DataArray,
    fill_value: Any,
) -> xr.Dataset:
    """
    Apply mask to all dataset variables except the mask band.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to mask.
    mask_band : str
        Name of mask band to exclude from masking.
    mask : xr.DataArray
        Boolean mask array (True = keep, False = mask).
    fill_value : Any
        Value to use for masked pixels.

    Returns
    -------
    xr.Dataset
        Dataset with masked variables.
    """
    result = dataset.copy()
    for var in dataset.data_vars:
        if var == mask_band:
            continue
        result[var] = dataset[var].where(mask, fill_value)
    return result


def _apply_mask_statistics(
    dataset: xr.Dataset,
    mask: xr.DataArray,
    intersects: str | dict[str, Any] | gpd.GeoDataFrame | None,
    bbox: list | tuple | None,
    bbox_crs: str,
) -> xr.Dataset:
    """
    Compute and add mask statistics to dataset coordinates.

    Calculates clear pixel counts and clear coverage percentages, optionally within
    a specified geometry. Adds clear_pixels (int32) and clear_percent (float32, 0-100)
    coordinates to the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to add statistics to.
    mask : xr.DataArray
        Boolean mask array for statistics calculation.
    intersects : str | dict[str, Any] | gpd.GeoDataFrame | None
        Optional geometry to compute statistics within.
    bbox : list | tuple | None
        Optional bounding box [min_x, min_y, max_x, max_y] for statistics.
    bbox_crs : str
        CRS for bbox parameter.

    Returns
    -------
    xr.Dataset
        Dataset with clear_pixels and clear_percent coordinates added.
    """
    spatial_dims = [d for d in mask.dims if d in [DIM_X, DIM_Y]]
    if not spatial_dims:
        spatial_dims = [d for d in mask.dims if d != DIM_TIME]
    if not spatial_dims:
        return dataset
    usable_pixels = dataset.attrs.get("usable_pixels")
    clip_mask_arr = None
    geometry_selection = _resolve_geometry_selection(intersects, bbox, bbox_crs)
    if usable_pixels is None and geometry_selection is not None:
        gdf = geometry_selection.to_crs(dataset.rio.crs)
        clip_mask_arr = geometry_mask(
            geometries=gdf.geometry,
            out_shape=(int(dataset.rio.height), int(dataset.rio.width)),
            transform=dataset.rio.transform(recalc=True),
            all_touched=False,
        )
        usable_pixels = int(np.sum(~clip_mask_arr))
        dataset.attrs["usable_pixels"] = usable_pixels
    if usable_pixels is None:
        usable_pixels = int(np.prod([mask.sizes[d] for d in spatial_dims]))
        dataset.attrs["usable_pixels"] = usable_pixels
    mask_for_stats = mask
    if clip_mask_arr is not None:
        if len(spatial_dims) < 2:
            raise DatacubeMaskingError(
                f"Expected at least 2 spatial dimensions for geometry clipping, got {spatial_dims}"
            )
        spatial_dims_str = [str(d) for d in spatial_dims]
        y_dim, x_dim = _infer_spatial_dims(dataset, spatial_dims_str)
        geometry_clip_xr = xr.DataArray(
            ~clip_mask_arr,
            dims=[y_dim, x_dim],
            coords={y_dim: dataset[y_dim], x_dim: dataset[x_dim]},
        )
        mask_for_stats = mask & geometry_clip_xr
    clear_pixels = mask_for_stats.sum(dim=tuple(spatial_dims))
    if usable_pixels > 0:
        clear_cover_pct = (clear_pixels / usable_pixels) * PERCENT_MAX
        clear_cover_pct = clear_cover_pct.clip(PERCENT_MIN, PERCENT_MAX).fillna(PERCENT_MIN)
    else:
        clear_cover_pct = xr.zeros_like(clear_pixels, dtype=np.float32)
    dataset.coords["clear_pixels"] = clear_pixels.astype(np.int32)
    dataset.coords["clear_percent"] = clear_cover_pct.astype(np.float32)
    return dataset


def _resolve_geometry_selection(
    intersects: str | dict[str, Any] | gpd.GeoDataFrame | None,
    bbox: list | tuple | None,
    bbox_crs: str,
) -> gpd.GeoDataFrame | None:
    """
    Resolve geometry selection from various input formats.

    Converts intersects or bbox parameters to a GeoDataFrame with proper CRS handling.
    Bbox takes precedence if both are provided.

    Parameters
    ----------
    intersects : str | dict[str, Any] | gpd.GeoDataFrame | None
        Geometry as WKT string, GeoJSON dict, or GeoDataFrame.
    bbox : list | tuple | None
        Bounding box [min_x, min_y, max_x, max_y].
    bbox_crs : str
        CRS to use for bbox or as fallback for intersects.

    Returns
    -------
    gpd.GeoDataFrame | None
        GeoDataFrame with geometry, or None if no geometry provided.
    """
    if bbox is not None:
        return bbox_to_geopandas(bbox, crs=bbox_crs)
    if intersects is None:
        return None
    if isinstance(intersects, gpd.GeoDataFrame):
        return intersects if intersects.crs is not None else intersects.set_crs(bbox_crs)
    gdf = geometry_to_geopandas(intersects)
    if gdf.crs is None:
        gdf = gdf.set_crs(bbox_crs)
    return gdf


def _infer_spatial_dims(dataset: xr.Dataset, spatial_dims: list[str]) -> tuple[str, str]:
    """
    Infer y and x dimension names from dataset.

    Attempts to use rioxarray metadata, falls back to assuming first two spatial
    dimensions are (y, x) order.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with spatial dimensions.
    spatial_dims : list[str]
        List of spatial dimension names.

    Returns
    -------
    tuple[str, str]
        (y_dim, x_dim) tuple.

    Warns
    -----
    UserWarning
        If rioxarray metadata is unavailable and fallback is used.
    """
    try:
        return dataset.rio.y_dim, dataset.rio.x_dim
    except Exception:
        warnings.warn(
            f"Could not determine spatial dimension names from rioxarray metadata. "
            f"Assuming spatial_dims {spatial_dims[:2]} are in (y, x) order. "
            f"If geometry clipping produces incorrect results, ensure dataset has rioxarray spatial metadata.",
            UserWarning,
            stacklevel=2,
        )
        return spatial_dims[0], spatial_dims[1]


def _filter_by_clear_cover(dataset: xr.Dataset, clear_cover: float) -> xr.Dataset:
    """
    Filter dataset by minimum clear coverage percentage threshold.

    Removes time steps (or entire dataset) that don't meet the minimum clear_cover
    threshold. Requires clear_percent coordinate from mask_statistics.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with clear_percent coordinate.
    clear_cover : float
        Minimum clear coverage percentage (0-100) threshold.

    Returns
    -------
    xr.Dataset
        Filtered dataset with only time steps meeting the threshold.

    Raises
    ------
    DatacubeMaskingError
        If clear_percent coordinate is missing, dataset doesn't meet threshold,
        or clear_percent has unsupported dimensionality.
    """
    if "clear_percent" not in dataset.coords:
        raise DatacubeMaskingError("clear_cover filtering requires mask_statistics=True")
    dims = dataset["clear_percent"].dims
    if len(dims) == 0:
        if float(dataset["clear_percent"].values) < clear_cover:
            raise DatacubeMaskingError(f"Dataset does not meet minimum clear_cover threshold of {clear_cover}%")
        return dataset
    if len(dims) == 1:
        dim_name = dims[0]
        mask = (dataset["clear_percent"] >= clear_cover).compute()
        return dataset.isel({dim_name: mask})
    raise DatacubeMaskingError(f"clear_cover filtering not supported for multi-dimensional clear_percent: {dims}")
