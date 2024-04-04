# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:32:31 2023

@author: nkk
"""

from rasterio import features
import numpy as np
import xarray as xr
import tqdm

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


def zonal_stats_numpy(
    dataset,
    gdf,
    operations=dict(mean=np.nanmean),
    all_touched=False,
    preload_datavar=False,
):
    tqdm_bar = tqdm.tqdm(total=len(dataset.data_vars) * dataset.time.size)
    dataset = dataset.rio.clip_box(*gdf.to_crs(dataset.rio.crs).total_bounds)

    feats, yx_pos = _rasterize(gdf, dataset, all_touched=all_touched)
    ds = []
    features_idx = []
    for data_var in dataset.data_vars:
        tqdm_bar.set_description(data_var)
        dataset_var = dataset[data_var]
        if preload_datavar:
            dataset_var = dataset_var.load()
        vals = {}
        for t in range(dataset_var.time.size):
            tqdm_bar.update(1)
            vals[t] = []
            mem_asset = dataset_var.isel(time=t).to_numpy()
            for i in range(gdf.shape[0]):
                features_idx.append(i)
                if len(yx_pos) <= i + 1:
                    break
                pos = np.asarray(yx_pos[i + 1])
                # mem_asset[*pos] only for python>=3.11
                if len(pos) == 2:
                    data = mem_asset[pos[0], pos[1]]
                elif len(pos) == 1:
                    data = mem_asset[pos[0]]
                if data.size > 0:
                    res = [operation(data) for operation in operations.values()]
                else:
                    res = [np.nan for operation in operations]
                vals[t].append(res)
        arr = np.asarray([vals[v] for v in vals])

        da = xr.DataArray(
            arr,
            dims=["time", "feature", "stats"],
            coords=dict(
                time=dataset_var.time.values,
                feature=gdf.index[np.nonzero(np.unique(feats))[0] - 1],
                stats=list(operations.keys()),
            ),
        )
        del arr, mem_asset, vals, dataset_var
        ds.append(da.to_dataset(name=data_var))
    tqdm_bar.close()
    return xr.merge(ds).transpose("feature", "time", "stats")


def zonal_stats(
    dataset,
    gdf,
    operations: list = ["mean"],
    all_touched=False,
    method="geocube",
    verbose=False,
    raise_missing_geometry=False,
):
    """


    Parameters
    ----------
    dataset : xr.Dataset
        DESCRIPTION.
    gdf : gpd.GeoDataFrame
        DESCRIPTION.
    operations : TYPE, list.
        DESCRIPTION. The default is ["mean"].
    all_touched : TYPE, optional
        DESCRIPTION. The default is False.
    method : TYPE, optional
        DESCRIPTION. The default is "geocube".
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    raise_missing_geometry : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    ValueError
        DESCRIPTION.
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if method == "geocube":
        from geocube.api.core import make_geocube
        from geocube.rasterize import rasterize_image

        def custom_rasterize_image(all_touched=all_touched, **kwargs):
            return rasterize_image(all_touched=all_touched, **kwargs)

        gdf["tmp_index"] = np.arange(gdf.shape[0])
        out_grid = make_geocube(
            gdf,
            measurements=["tmp_index"],
            like=dataset,  # ensure the data are on the same grid
            rasterize_function=custom_rasterize_image,
        )
        cube = dataset.groupby(out_grid.tmp_index)
        zonal_stats = xr.concat(
            [getattr(cube, operation)() for operation in operations], dim="stats"
        )
        zonal_stats["stats"] = operations

        if zonal_stats["tmp_index"].size != gdf.shape[0]:
            index_list = [
                gdf.index[i] for i in zonal_stats["tmp_index"].values.astype(np.int16)
            ]
            if raise_missing_geometry:
                diff = gdf.shape[0] - len(index_list)
                raise ValueError(
                    f'{diff} geometr{"y is" if diff==1 else "ies are"} missing in the zonal stats. This can be due to too small geometries, duplicated...'
                )
        else:
            index_list = list(gdf.index)
        zonal_stats["tmp_index"] = index_list
        return zonal_stats.rename(dict(tmp_index="feature"))

    tqdm_bar = tqdm.tqdm(total=gdf.shape[0])

    if dataset.rio.crs != gdf.crs:
        Warning(
            f"Different projections. Reproject vector to EPSG:{dataset.rio.crs.to_epsg()}."
        )
        gdf = gdf.to_crs(dataset.rio.crs)

    zonal_ds_list = []

    dataset = dataset.rio.clip_box(*gdf.to_crs(dataset.rio.crs).total_bounds)

    if method == "optimized":
        feats, yx_pos = _rasterize(gdf, dataset, all_touched=all_touched)

        for gdf_idx in tqdm.trange(gdf.shape[0], disable=not verbose):
            tqdm_bar.update(1)
            if gdf_idx + 1 >= len(yx_pos):
                continue
            yx_pos_idx = yx_pos[gdf_idx + 1]
            if np.asarray(yx_pos_idx).size == 0:
                continue
            datacube_spatial_subset = dataset.isel(
                x=xr.DataArray(yx_pos_idx[1], dims="xy"),
                y=xr.DataArray(yx_pos_idx[0], dims="xy"),
            )
            del yx_pos_idx
            zonal_ds_list.append(
                datacube_time_stats(datacube_spatial_subset, operations).expand_dims(
                    dim={"feature": [gdf.iloc[gdf_idx].name]}
                )
            )

        del yx_pos, feats

    elif method == "standard":
        for idx_gdb, feat in tqdm.tqdm(
            gdf.iterrows(), total=gdf.shape[0], disable=not verbose
        ):
            tqdm_bar.update(1)
            if feat.geometry.geom_type == "MultiPolygon":
                shapes = feat.geometry.geoms
            else:
                shapes = [feat.geometry]
            datacube_spatial_subset = dataset.rio.clip(shapes, all_touched=all_touched)

            zonal_feat = datacube_time_stats(
                datacube_spatial_subset, operations
            ).expand_dims(dim={"feature": [feat.name]})

            zonal_ds_list.append(zonal_feat)
    else:
        raise NotImplementedError('method available are : "standard" or "optimized"')
    return xr.concat(zonal_ds_list, dim="feature")
