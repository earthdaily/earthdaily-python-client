from typing import Any

import geopandas as gpd
import numpy as np
import rasterio.features
import xarray as xr
from scipy.stats import mode
from shapely import wkt

from earthdaily.datacube._geometry import geometry_to_geopandas
from earthdaily.datacube.constants import DEFAULT_ZONAL_REDUCERS, DIM_FEATURE, DIM_X, DIM_Y, DIM_ZONAL_STATS


def compute_zonal_stats(
    dataset: xr.Dataset,
    geometry: str | dict[str, Any] | gpd.GeoDataFrame,
    reducers: list[str] = DEFAULT_ZONAL_REDUCERS,
    all_touched: bool = True,
    preserve_columns: bool = True,
    lazy_load: bool = True,
) -> xr.Dataset:
    """
    Compute zonal statistics for geometries over a dataset.

    Calculates statistical summaries (mean, median, mode, etc.) for each geometry feature
    by rasterizing geometries onto the dataset grid and aggregating pixel values within
    each geometry. Supports multiple reducer functions and preserves geometry attributes.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset with spatial dimensions (x, y). Must have rioxarray spatial metadata.
    geometry : str | dict[str, Any] | gpd.GeoDataFrame
        Geometries to compute statistics for. Can be WKT string, GeoJSON dict, or
        GeoDataFrame. Will be reprojected to match dataset CRS.
    reducers : list[str], optional
        List of statistical reducer functions to apply. Supported: "mean", "median",
        "mode", "std", "min", "max", "sum", "count", and other numpy functions.
        Default is DEFAULT_ZONAL_REDUCERS (["mean"]).
    all_touched : bool, optional
        Whether to include pixels touched by geometry boundaries in rasterization.
        Default is True.
    preserve_columns : bool, optional
        Whether to preserve non-geometry columns from the input GeoDataFrame as
        coordinates in the output. Default is True.
    lazy_load : bool, optional
        Whether to return a lazy dask-backed Dataset or compute immediately. Default is True.

    Returns
    -------
    xr.Dataset
        Dataset with zonal statistics. Has dimensions (feature, zonal_statistics) where
        feature corresponds to input geometries and zonal_statistics corresponds to reducers.
        Includes geometry coordinates as WKT strings and optionally preserved columns.

    Raises
    ------
    ValueError
        If no valid geometry positions are found after rasterization.
    """
    geometry_gdf = geometry_to_geopandas(geometry)
    return _compute_zonal_statistics(
        dataset,
        geometry_gdf,
        reducers=reducers,
        all_touched=all_touched,
        preserve_columns=preserve_columns,
        lazy_load=lazy_load,
    )


def _compute_zonal_statistics(
    dataset: xr.Dataset,
    geometries: gpd.GeoDataFrame,
    reducers: list[str] = DEFAULT_ZONAL_REDUCERS,
    all_touched: bool = True,
    preserve_columns: bool = True,
    lazy_load: bool = True,
) -> xr.Dataset:
    """
    Internal function to compute zonal statistics.

    Reprojects geometries to dataset CRS, clips dataset to geometry bounds, rechunks
    spatial dimensions to ensure entire arrays are loaded for rasterization, then computes
    statistics for each geometry.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset with spatial dimensions.
    geometries : gpd.GeoDataFrame
        Geometries to compute statistics for.
    reducers : list[str], optional
        List of statistical reducer function names. Default is DEFAULT_ZONAL_REDUCERS.
    all_touched : bool, optional
        Whether to include pixels touched by geometry boundaries in rasterization.
        Default is True.
    preserve_columns : bool, optional
        Whether to preserve GeoDataFrame columns. Default is True.
    lazy_load : bool, optional
        Whether to return lazy dask-backed Dataset. Default is True.

    Returns
    -------
    xr.Dataset
        Dataset with zonal statistics computed.

    Raises
    ------
    ValueError
        If no valid geometry positions are found after rasterization.
    """
    geometries = geometries.to_crs(dataset.rio.crs)
    dataset = dataset.rio.clip_box(*geometries.total_bounds)

    # Rechunk spatial dimensions to single chunks so rasterio can access full transform/shape
    if DIM_Y in dataset.dims and DIM_X in dataset.dims:
        dataset = dataset.chunk({DIM_Y: -1, DIM_X: -1})

    features, positions_raw = _rasterize_geometries(geometries, dataset, all_touched)
    positions = [pos for pos in positions_raw if len(pos) > 0]

    if len(positions) == 0:
        raise ValueError("No valid geometry positions found after rasterization")

    stats_ds = _compute_stats(dataset, positions, reducers)
    formatted = _format_output(stats_ds, features, geometries, preserve_columns)
    return formatted if lazy_load else formatted.compute()


def _rasterize_geometries(
    gdf: gpd.GeoDataFrame, dataset: xr.Dataset, all_touched: bool
) -> tuple[list[dict[str, Any]], list[Any]]:
    """
    Rasterize geometries onto the dataset grid.

    Converts vector geometries to raster masks and extracts pixel coordinates for each
    geometry. Returns features and their corresponding pixel positions for statistics
    computation.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Geometries to rasterize.
    dataset : xr.Dataset
        Dataset to rasterize onto. Used for transform and shape.
    all_touched : bool
        Whether to include pixels touched by geometry boundaries.

    Returns
    -------
    tuple[list[dict[str, Any]], list[Any]]
        Tuple of (features, positions) where features are dicts with geometry info and
        positions are tuples of (y_coords, x_coords) arrays for each geometry. Empty
        positions indicate geometries that don't intersect the raster.
    """
    transform = dataset.rio.transform()
    shape = (dataset.rio.height, dataset.rio.width)

    geom_list = [(geom, idx) for idx, geom in enumerate(gdf.geometry)]
    raster = rasterio.features.rasterize(
        geom_list,
        out_shape=shape,
        transform=transform,
        all_touched=all_touched,
        fill=-1,
        dtype=np.int32,
    )

    features: list[dict[str, Any]] = []
    positions: list[Any] = []

    for idx in range(len(gdf)):
        mask = raster == idx
        if mask.any():
            y_coords, x_coords = np.where(mask)
            positions.append((y_coords, x_coords))
            features.append({"geometry": gdf.iloc[idx].geometry, "index": idx})
        else:
            positions.append(())

    return features, positions


def _compute_stats(dataset: xr.Dataset, positions: list[Any], reducers: list[str]) -> xr.Dataset:
    """
    Compute statistical summaries for each geometry's pixel positions.

    Applies reducer functions to pixel values within each geometry. Supports numpy
    statistical functions (mean, median, std, min, max, sum, etc.) and scipy.stats.mode.
    Uses apply_ufunc for efficient parallel computation across all variables and time steps.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to compute statistics from.
    positions : list[Any]
        List of (y_coords, x_coords) tuples for each geometry.
    reducers : list[str]
        List of reducer function names to apply.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (feature, zonal_statistics) containing computed statistics
        for each variable and reducer combination.
    """

    def _zonal_stats_ufunc(data: np.ndarray, positions: list[Any], reducers: list[str]) -> np.ndarray:
        results = []
        for pos in positions:
            field_stats = []
            for reducer in reducers:
                if len(pos) == 2:
                    field_arr = data[(...,) + tuple(pos)]
                else:
                    field_arr = np.array([])

                if reducer == "mode":
                    if field_arr.size > 0:
                        field_val = mode(field_arr, axis=-1, nan_policy="omit").mode
                    else:
                        field_val = np.nan
                else:
                    func_name = f"nan{reducer}" if hasattr(np, f"nan{reducer}") else reducer
                    if field_arr.size > 0:
                        field_val = getattr(np, func_name)(field_arr, axis=-1)
                    else:
                        field_val = np.nan

                field_stats.append(field_val)

            results.append(np.asarray(field_stats))

        results_array = np.asarray(results)
        return results_array.swapaxes(-1, 0).swapaxes(-1, -2)

    result = xr.apply_ufunc(
        _zonal_stats_ufunc,
        dataset,
        input_core_dims=[[DIM_Y, DIM_X]],
        output_core_dims=[[DIM_FEATURE, DIM_ZONAL_STATS]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={"positions": positions, "reducers": reducers},
        dask_gufunc_kwargs={"output_sizes": {DIM_FEATURE: len(positions), DIM_ZONAL_STATS: len(reducers)}},
    )

    result = result.assign_coords({DIM_ZONAL_STATS: reducers})
    return result


def _format_output(
    stats: xr.Dataset,
    features: list[dict[str, Any]],
    geometries: gpd.GeoDataFrame,
    preserve_columns: bool,
) -> xr.Dataset:
    """
    Format output dataset with geometry and attribute coordinates.

    Adds geometry WKT strings and optionally preserves GeoDataFrame columns as
    coordinates in the output dataset.

    Parameters
    ----------
    stats : xr.Dataset
        Dataset with computed statistics.
    features : list[dict[str, Any]]
        List of feature dictionaries containing geometry information.
    geometries : gpd.GeoDataFrame
        Original geometries GeoDataFrame for column preservation.
    preserve_columns : bool
        Whether to preserve non-geometry columns from geometries.

    Returns
    -------
    xr.Dataset
        Formatted dataset with geometry and optional attribute coordinates.
    """
    geometry_wkt = [wkt.dumps(f["geometry"]) for f in features]
    stats = stats.assign_coords({"geometry": (DIM_FEATURE, geometry_wkt)})
    feature_indices = [f.get("index", idx) for idx, f in enumerate(features)]

    if preserve_columns:
        if feature_indices:
            valid_rows = geometries.iloc[feature_indices]
        else:
            valid_rows = geometries.iloc[[]]

        for col in geometries.columns:
            if col != "geometry":
                values = valid_rows[col].values
                stats = stats.assign_coords({col: (DIM_FEATURE, values)})

    return stats
