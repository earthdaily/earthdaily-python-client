# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:32:31 2023

@author: nkk
"""

from rasterio import features
import numpy as np
import xarray as xr
import tqdm
import logging
import time
from . import custom_reducers
from .preprocessing import rasterize
from scipy.sparse import csr_matrix


def _compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def _indices_sparse(data):
    M = _compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def _np_mode(arr, **kwargs):
    values, counts = np.unique(arr, return_counts=True)
    isnan = np.isnan(values)
    values, counts = values[~isnan], counts[~isnan]
    return values[np.argmax(counts)]


def datacube_time_stats(datacube, operations):
    datacube = datacube.groupby("time")
    stats = []
    for operation in operations:
        stat = getattr(datacube, operation)(...)
        stats.append(stat.expand_dims(dim={"stats": [operation]}))
    stats = xr.concat(stats, dim="stats")
    return stats


def _rasterize(gdf, dataset, all_touched=False):
    feats = rasterize(gdf, dataset, all_touched=all_touched)
    yx_pos = _indices_sparse(feats)
    return feats, yx_pos


def _memory_time_chunks(dataset, memory=None):
    import psutil

    if memory is None:
        memory = psutil.virtual_memory().available / 1e6
        logging.debug(f"Hoping to use a maximum memory {memory}Mo.")
    nbytes_per_date = int(dataset.nbytes / 1e6) / dataset.time.size * 3
    max_time_chunks = int(np.arange(0, memory, nbytes_per_date + 0.1).size)
    time_chunks = int(
        dataset.time.size / np.arange(0, dataset.time.size, max_time_chunks).size
    )
    logging.debug(
        f"Mo per date : {nbytes_per_date:0.2f}, total : {(nbytes_per_date*dataset.time.size):0.2f}."
    )
    logging.debug(f"Time chunks : {time_chunks} (on {dataset.time.size} time).")
    return time_chunks


def _zonal_stats_numpy(
    dataset: xr.Dataset, positions, reducers: list = ["mean"], all_touched=False
):
    def _zonal_stats_ufunc(dataset, positions, reducers):
        zs = []
        for idx in range(len(positions)):
            field_stats = []
            for reducer in reducers:
                field_arr = dataset[(...,) + tuple(positions[idx])]
                func = f"nan{reducer}" if hasattr(np, f"nan{reducer}") else reducer
                field_arr = getattr(np, func)(field_arr, axis=-1)
                field_stats.append(field_arr)
            field_stats = np.asarray(field_stats)
            zs.append(field_stats)
        zs = np.asarray(zs)
        zs = zs.swapaxes(-1, 0).swapaxes(-1, -2)
        return zs

    dask_ufunc = "parallelized"

    zs = xr.apply_ufunc(
        _zonal_stats_ufunc,
        dataset,
        vectorize=False,
        dask=dask_ufunc,
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

    del dataset

    return zs


def zonal_stats(
    dataset: xr.Dataset,
    geoms,
    method: str = "numpy",
    smart_load: bool = False,
    memory: int = None,
    reducers: list = ["mean"],
    all_touched=True,
    label=None,
    buffer_meters: int | float | None = None,
    **kwargs,
):
    """
    Xr Zonal stats using np.nan functions.

    Parameters
    ----------
    dataset : xr.Dataset
        DESCRIPTION.
    geoms : TYPE
        DESCRIPTION.
    method : str
        "xvec" or "numpy". The default is "numpy".
    smart_load : bool
        Will load in memory the maximum of time and loop on it for "numpy"
        method. The default is False.
    memory : int, optional
        Only for the "numpy" method, by default it will take the maximum memory
        available. But in some cases it can be too much or too little.
        The default is None.
    reducers : list, optional
        Any np.nan function ("mean" is "np.nanmean"). The default is ['mean'].

    Yields
    ------
    zs : TYPE
        DESCRIPTION.

    """

    def _loop_time_chunks(dataset, method, smart_load, time_chunks):
        logging.debug(
            f"Batching every {time_chunks} dates ({np.ceil(dataset.time.size/time_chunks).astype(int)} loops)."
        )
        for time_idx in tqdm.trange(0, dataset.time.size, time_chunks):
            isel_time = np.arange(
                time_idx, np.min((time_idx + time_chunks, dataset.time.size))
            )
            ds = dataset.copy().isel(time=isel_time)
            if smart_load:
                t0 = time.time()
                ds = ds.load()
                logging.debug(
                    f"Subdataset of {ds.time.size} dates loaded in memory in {(time.time()-t0):0.2f}s."
                )
            t0 = time.time()
            # for method in tqdm.tqdm(["np"]):
            zs = _zonal_stats_numpy(ds, positions, reducers)
            zs = zs.load()
            del ds
            logging.debug(f"Zonal stats computed in {(time.time()-t0):0.2f}s.")
            yield zs

    t_start = time.time()
    dataset = dataset.rio.clip_box(*geoms.to_crs(dataset.rio.crs).total_bounds)
    if isinstance(buffer_meters, float | int):
        input_crs = geoms.crs
        geoms = geoms.to_crs({"proj": "cea"})
        geoms["geometry_original"] = geoms.geometry
        geoms.geometry = geoms.buffer(buffer_meters)
        geoms.to_crs(input_crs)

    if method == "numpy":
        feats, yx_pos = _rasterize(geoms.copy(), dataset, all_touched=all_touched)
        positions = [np.asarray(yx_pos[i + 1]) for i in np.arange(geoms.shape[0])]
        positions = [position for position in positions if position.size > 0]
        del yx_pos
        time_chunks = _memory_time_chunks(dataset, memory)
        if smart_load:
            zs = xr.concat(
                [
                    z
                    for z in _loop_time_chunks(dataset, method, smart_load, time_chunks)
                ],
                dim="time",
            )
        else:
            zs = _zonal_stats_numpy(dataset, positions, reducers, **kwargs)
        zs = zs.assign_coords(zonal_statistics=reducers)
        zs = zs.rio.write_crs("EPSG:4326")

        # keep only geom that have been found in the raster
        f = np.unique(feats)
        f = f[f > 0]
        index = geoms.index[f - 1]

        index = xr.DataArray(
            index, dims=["feature"], coords={"feature": zs.feature.values}
        )

        # create the WKT geom
        if isinstance(buffer_meters, float | int):
            geoms.geometry = geoms["geometry_original"]

        if geoms.crs.to_epsg() != 4326:
            geoms = geoms.to_crs("EPSG:4326")
        geometry = xr.DataArray(
            geoms.iloc[list(f - 1)].geometry.apply(lambda x: x.wkt).values,
            dims=["feature"],
            coords={"feature": zs.feature.values},
        )
        new_coords_kwargs = {"index": index, "geometry": geometry}

        # add the label if a column is specified
        if label:
            label = xr.DataArray(
                list(geoms[label].iloc[f - 1]),
                dims=["feature"],
                coords={"feature": zs.feature.values},
            )
            new_coords_kwargs["label"] = label

        zs = zs.assign_coords(**new_coords_kwargs)
        zs = zs.set_index(feature=list(new_coords_kwargs.keys()))

    if method == "xvec":
        import xvec

        zs = dataset.xvec.zonal_stats(
            geoms.to_crs(dataset.rio.crs).geometry,
            y_coords="y",
            x_coords="x",
            stats=reducers,
            method="rasterize",
            all_touched=all_touched,
            **kwargs,
        )
    logging.info(f"Zonal stats method {method} tooks {time.time()-t_start}s.")
    del dataset
    return zs
