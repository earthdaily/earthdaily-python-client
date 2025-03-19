"""Zonal statistics computation for geospatial data analysis.

This module provides functionality for computing zonal statistics on xarray Datasets
using various methods including numpy and xvec. It supports parallel processing,
memory optimization, and various statistical operations.

Example:
    >>> import xarray as xr
    >>> import geopandas as gpd
    >>> dataset = xr.open_dataset("temperature.nc")
    >>> polygons = gpd.read_file("zones.geojson")
    >>> stats = zonal_stats(dataset, polygons, reducers=["mean", "max"])
"""

import logging
import time
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import psutil
import xarray as xr
from scipy.sparse import csr_matrix
from scipy.stats import mode
from tqdm.auto import trange

from .preprocessing import rasterize

# Configure logging
logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory allocation for large dataset processing."""

    @staticmethod
    def calculate_time_chunks(
        dataset: xr.Dataset, max_memory_mb: Optional[float] = None
    ) -> int:
        """Calculate optimal time chunks based on available memory.

        Args:
            dataset: Input xarray Dataset
            max_memory_mb: Maximum memory to use in megabytes

        Returns:
            int: Optimal number of time chunks
        """
        if max_memory_mb is None:
            max_memory_mb = psutil.virtual_memory().available / 1e6
            logger.info(f"Using maximum available memory: {max_memory_mb:.2f}MB")

        bytes_per_date = (dataset.nbytes / 1e6) / dataset.time.size * 3
        max_chunks = int(np.arange(0, max_memory_mb, bytes_per_date + 0.1).size)
        time_chunks = int(
            dataset.time.size / np.arange(0, dataset.time.size, max_chunks).size
        )

        logger.info(
            f"Estimated memory per date: {bytes_per_date:.2f}MB. Total: {(bytes_per_date * dataset.time.size):.2f}MB"
        )
        logger.info(
            f"Time chunks: {time_chunks} (total time steps: {dataset.time.size})"
        )

        return time_chunks


class SpatialIndexer:
    """Handles spatial indexing and rasterization operations."""

    @staticmethod
    def compute_sparse_matrix(data: np.ndarray) -> csr_matrix:
        """Compute sparse matrix from input data.

        Args:
            data: Input numpy array

        Returns:
            scipy.sparse.csr_matrix: Computed sparse matrix
        """
        cols = np.arange(data.size)
        return csr_matrix(
            (cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size)
        )

    @staticmethod
    def get_sparse_indices(data: np.ndarray) -> List[Tuple[np.ndarray, ...]]:
        """Get sparse indices from input data.

        Args:
            data: Input numpy array

        Returns:
            List of index tuples
        """
        matrix = SpatialIndexer.compute_sparse_matrix(data)
        return [np.unravel_index(row.data, data.shape) for row in matrix]

    @staticmethod
    def rasterize_geometries(
        gdf: gpd.GeoDataFrame, dataset: xr.Dataset, all_touched: bool = False
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Rasterize geometries to match dataset resolution.

        Args:
            gdf: Input GeoDataFrame
            dataset: Reference dataset for rasterization
            all_touched: Whether to include all touched pixels

        Returns:
            Tuple containing features array and positions list
        """
        features = rasterize(gdf, dataset, all_touched=all_touched)
        positions = SpatialIndexer.get_sparse_indices(features)
        return features, positions


class StatisticalOperations:
    """Handles statistical computations on spatial data."""

    @staticmethod
    def zonal_stats(
        dataset: xr.Dataset, positions: List[np.ndarray], reducers: List[str]
    ) -> xr.DataArray:
        """Compute zonal statistics for given positions using specified reducers.

        Args:
            dataset: Input dataset
            positions: List of position arrays
            reducers: List of statistical operations to perform

        Returns:
            xarray.DataArray: Computed statistics

        Notes:
            Uses xarray's apply_ufunc for parallel processing and efficient computation
        """

        def _zonal_stats_ufunc(data, positions, reducers):
            """Inner function for parallel computation of zonal statistics."""
            zs = []
            for idx in range(len(positions)):
                field_stats = []
                for reducer in reducers:
                    field_arr = data[(...,) + tuple(positions[idx])]
                    if reducer == "mode":
                        field_arr = mode(field_arr, axis=-1, nan_policy="omit").mode
                    else:
                        func = (
                            f"nan{reducer}" if hasattr(np, f"nan{reducer}") else reducer
                        )
                        field_arr = getattr(np, func)(field_arr, axis=-1)
                    field_stats.append(field_arr)
                field_stats = np.asarray(field_stats)
                zs.append(field_stats)
            zs = np.asarray(zs)
            return zs.swapaxes(-1, 0).swapaxes(-1, -2)

        # Apply the function using xarray's parallel processing capabilities
        return xr.apply_ufunc(
            _zonal_stats_ufunc,
            dataset,
            vectorize=False,
            dask="parallelized",
            input_core_dims=[["y", "x"]],
            output_core_dims=[["feature", "zonal_statistics"]],
            exclude_dims=set(["x", "y"]),
            output_dtypes=[float],
            kwargs=dict(reducers=reducers, positions=positions),
            dask_gufunc_kwargs={
                "allow_rechunk": True,
                "output_sizes": dict(
                    feature=len(positions), zonal_statistics=len(reducers)
                ),
            },
        )


def zonal_stats(
    dataset: xr.Dataset,
    geometries: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    method: str = "numpy",
    lazy_load: bool = True,
    max_memory_mb: Optional[float] = None,
    reducers: List[str] = ["mean"],
    all_touched: bool = True,
    preserve_columns: bool = True,
    buffer_meters: Optional[Union[int, float]] = None,
    **kwargs,
) -> xr.Dataset:
    """Calculate zonal statistics for xarray Dataset based on geometric boundaries.

    This function computes statistical summaries of Dataset values within each geometry's zone,
    supporting parallel processing through xarray's apply_ufunc and multiple computation methods.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset containing variables for statistics computation.
    geoms : Union[geopandas.GeoDataFrame, geopandas.GeoSeries]
        Geometries defining the zones for statistics calculation.
    method : str, optional
        Method for computation. Options:
            - 'numpy': Uses numpy functions with parallel processing
            - 'xvec': Uses xvec library (must be installed)
        Default is 'numpy'.
    lazy_load : bool, optional
        If True, optimizes memory usage by loading chunks of data for 'numpy' method.
        Default is False.
    max_memory_mb : float, optional
        Maximum memory to use in megabytes. If None, uses maximum available memory.
        Default is None.
    reducers : list[str], optional
        List of statistical operations to perform. Functions should be numpy nan-functions
        (e.g., 'mean' uses np.nanmean). Default is ['mean'].
    all_touched : bool, optional
        If True, includes all pixels touched by geometries in computation.
        Default is True.
    preserve_columns : bool, optional
        If True, preserves all columns from input geometries in output.
        Default is True.
    buffer_meters : Union[int, float, None], optional
        Buffer distance in meters to apply to geometries before computation.
        Default is None.
    **kwargs : dict
        Additional keyword arguments passed to underlying computation functions.

    Returns
    -------
    xarray.Dataset
        Dataset containing computed statistics with dimensions:
            - time (if present in input)
            - feature (number of geometries)
            - zonal_statistics (number of reducers)
        Additional coordinates include geometry WKT and preserved columns if requested.

    See Also
    --------
    xarray.apply_ufunc : Function used for parallel computation
    rasterio.features : Used for geometry rasterization

    Notes
    -----
    Memory usage is optimized for time series data when lazy_load=True by processing
    in chunks determined by available system memory.

    The 'xvec' method requires the xvec package to be installed separately.

    Examples
    --------
    >>> import xarray as xr
    >>> import geopandas as gpd
    >>> dataset = xr.open_dataset("temperature.nc")
    >>> polygons = gpd.read_file("zones.geojson")
    >>> stats = compute_zonal_stats(
    ...     dataset,
    ...     polygons,
    ...     reducers=["mean", "max"],
    ...     lazy_load=True
    ... )

    Raises
    ------
    ImportError
        If 'xvec' method is selected but xvec package is not installed.
    ValueError
        If invalid method or reducer is specified.
    DeprecationWarning
        If deprecated parameters are used.
    """
    # Input validation and deprecation warnings
    if "label" in kwargs:
        raise DeprecationWarning(
            '"label" parameter is deprecated and removed in earthdaily>=0.5. '
            "All geometry columns are preserved by default (preserve_columns=True)."
        )

    if "smart_load" in kwargs:
        import warnings

        warnings.warn(
            '"smart_load" will be deprecated in earthdaily>=0.6. '
            'Use "lazy_load" instead (lazy_load=True == smart_load=False).',
            DeprecationWarning,
        )
        lazy_load = not kwargs["smart_load"]

    # Clip dataset to geometry bounds
    dataset = dataset.rio.clip_box(*geometries.to_crs(dataset.rio.crs).total_bounds)

    # Apply buffer if specified
    if buffer_meters is not None:
        geometries = _apply_buffer(geometries, buffer_meters)

    if method == "numpy":
        return _compute_numpy_stats(
            dataset,
            geometries,
            lazy_load,
            max_memory_mb,
            reducers,
            all_touched,
            preserve_columns,
            **kwargs,
        )
    elif method == "xvec":
        return _compute_xvec_stats(
            dataset, geometries, reducers, all_touched, preserve_columns, **kwargs
        )
    else:
        raise ValueError(f"Unsupported method: {method}")


def _apply_buffer(
    geometries: gpd.GeoDataFrame, buffer_meters: Union[int, float]
) -> gpd.GeoDataFrame:
    """Apply buffer to geometries in meters."""
    original_crs = geometries.crs
    geometries = geometries.to_crs({"proj": "cea"})
    geometries["geometry_original"] = geometries.geometry
    geometries.geometry = geometries.buffer(buffer_meters)
    return geometries.to_crs(original_crs)


def _compute_numpy_stats(
    dataset: xr.Dataset,
    geometries: gpd.GeoDataFrame,
    lazy_load: bool,
    max_memory_mb: Optional[float],
    reducers: List[str],
    all_touched: bool,
    preserve_columns: bool,
    **kwargs,
) -> xr.Dataset:
    """Compute zonal statistics using numpy method."""
    # Rasterize geometries
    features, yx_positions = SpatialIndexer.rasterize_geometries(
        geometries.copy(), dataset, all_touched
    )
    positions = [np.asarray(pos) for pos in yx_positions[1:]]
    positions = [pos for pos in positions if pos.size > 0]

    # Process time series if present
    if "time" in dataset.dims and not lazy_load:
        time_chunks = MemoryManager.calculate_time_chunks(dataset, max_memory_mb)
        stats = _process_time_chunks(
            dataset, positions, reducers, lazy_load, time_chunks
        )
    else:
        stats = StatisticalOperations.zonal_stats(dataset, positions, reducers)

    # Format output
    return _format_numpy_output(stats, features, geometries, reducers, preserve_columns)


def _process_time_chunks(
    dataset: xr.Dataset,
    positions: List[np.ndarray],
    reducers: List[str],
    lazy_load: bool,
    time_chunks: int,
) -> xr.Dataset:
    """Process dataset in time chunks to optimize memory usage."""
    chunks = []
    for time_idx in trange(0, dataset.time.size, time_chunks):
        end_idx = min(time_idx + time_chunks, dataset.time.size)
        ds_chunk = dataset.isel(time=slice(time_idx, end_idx))

        if not lazy_load:
            load_start = time.time()
            ds_chunk = ds_chunk.load()
            logger.debug(
                f"Loaded {ds_chunk.time.size} dates in "
                f"{(time.time() - load_start):.2f}s"
            )

        compute_start = time.time()
        chunk_stats = StatisticalOperations.zonal_stats(ds_chunk, positions, reducers)
        logger.debug(
            f"Computed chunk statistics in {(time.time() - compute_start):.2f}s"
        )

        chunks.append(chunk_stats)

    return xr.concat(chunks, dim="time")


def _compute_xvec_stats(
    dataset: xr.Dataset,
    geometries: gpd.GeoDataFrame,
    reducers: List[str],
    all_touched: bool,
    preserve_columns: bool,
    **kwargs,
) -> xr.Dataset:
    """Compute zonal statistics using xvec method.

    Args:
        dataset: Input dataset
        geometries: Input geometries
        reducers: List of statistical operations to perform
        all_touched: Whether to include all touched pixels
        preserve_columns: Whether to preserve geometry columns
        **kwargs: Additional keyword arguments

    Returns:
        xarray.Dataset: Computed statistics

    Raises:
        ImportError: If xvec package is not installed
    """
    from importlib.util import find_spec

    if find_spec("xvec"):
        import xvec  # noqa: F401
    else:
        ImportError(
            "The xvec method requires the xvec package. "
            "Please install it with: pip install xvec"
        )

    # Compute statistics using xvec
    stats = dataset.xvec.zonal_stats(
        geometries.to_crs(dataset.rio.crs).geometry,
        y_coords="y",
        x_coords="x",
        stats=reducers,
        method="rasterize",
        all_touched=all_touched,
        **kwargs,
    )

    # Drop geometry and add as coordinate
    stats = stats.drop("geometry")
    stats = stats.assign_coords(
        geometry=("feature", geometries.geometry.to_wkt(rounding_precision=-1).values)
    )

    # Add index coordinate
    stats = stats.assign_coords(index=("feature", geometries.index))
    stats = stats.set_index(feature=["geometry", "index"])

    # Transpose dimensions to match numpy method output
    stats = stats.transpose("time", "feature", "zonal_statistics")

    # Preserve additional columns if requested
    if preserve_columns:
        stats = _preserve_geometry_columns(stats, geometries)

    return stats


def _format_numpy_output(
    stats: xr.Dataset,
    features: np.ndarray,
    geometries: gpd.GeoDataFrame,
    reducers: List[str],
    preserve_columns: bool,
) -> xr.Dataset:
    """Format numpy statistics output."""
    # Set coordinates and metadata
    stats = stats.assign_coords(zonal_statistics=reducers)
    stats = stats.rio.write_crs("EPSG:4326")

    # Process features and create index
    feature_indices = np.unique(features)
    feature_indices = feature_indices[feature_indices > 0]
    index = geometries.index[feature_indices - 1]

    # Convert geometries to WKT
    if geometries.crs.to_epsg() != 4326:
        geometries = geometries.to_crs("EPSG:4326")

    geometry_wkt = (
        geometries.geometry.iloc[feature_indices - 1]
        .to_wkt(rounding_precision=-1)
        .values
    )

    # Assign coordinates
    coords = {"index": ("feature", index), "geometry": ("feature", geometry_wkt)}
    stats = stats.assign_coords(**coords)
    stats = stats.set_index(feature=list(coords.keys()))

    # Preserve additional columns if requested
    if preserve_columns:
        stats = _preserve_geometry_columns(stats, geometries)

    return stats


def _preserve_geometry_columns(
    stats: xr.Dataset, geometries: gpd.GeoDataFrame
) -> xr.Dataset:
    """Preserve geometry columns in output statistics."""
    cols = [
        col for col in geometries.columns if col != geometries._geometry_column_name
    ]
    values = geometries.loc[stats.index.values][cols].values.T

    for col, val in zip(cols, values):
        stats = stats.assign_coords({col: ("feature", val)})

    feature_index = list(stats["feature"].to_index().names)
    feature_index.extend(cols)
    stats = stats.set_index(feature=feature_index)

    return stats
