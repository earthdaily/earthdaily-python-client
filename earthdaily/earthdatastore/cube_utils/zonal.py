# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:32:31 2023

@author: nkk
"""

from rasterio import features
from scipy.sparse import csr_matrix
import numpy as np
import xarray as xr
import tqdm
from . import custom_operations as _


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


def _zonal_stats_geom(datacube, operations):
    datacube = datacube.groupby("time")
    stats = []
    for operation in operations:
        stat = getattr(datacube, operation)(...)
        stats.append(stat.expand_dims(dim={"stats": [operation]}))
    stats = xr.concat(stats, dim="stats")
    return stats


def zonal_stats(
    dataset,
    gdf,
    operations=["mean"],
    all_touched=False,
    batch_rasterize=True,
    verbose=False,
):
    if dataset.rio.crs != gdf.crs:
        Warning(
            f"Different projections. Reproject vector to EPSG:{dataset.rio.crs.to_epsg()}."
        )
        gdf = gdf.to_crs(dataset.rio.crs)

    zonal_ds_list = []
    if batch_rasterize:
        shapes = ((gdf.iloc[i].geometry, i + 1) for i in range(gdf.shape[0]))

        # rasterize features to use numpy/scipy to avoid polygon clipping
        feats = features.rasterize(
            shapes=shapes,
            fill=0,
            out_shape=dataset.rio.shape,
            transform=dataset.rio.transform(),
            all_touched=all_touched,
        )

        idx_start = 0
        if 0 in feats:
            idx_start = 1
        yx_pos = _indices_sparse(feats)

        for gdf_idx in tqdm.trange(gdf.shape[0], disable=not verbose):
            yx_pos_idx = yx_pos[gdf_idx + idx_start]
            datacube_spatial_subset = dataset.isel(
                x=xr.DataArray(yx_pos_idx[1], dims="xy"),
                y=xr.DataArray(yx_pos_idx[0], dims="xy"),
            )
            del yx_pos_idx
            zonal_ds_list.append(
                _zonal_stats_geom(datacube_spatial_subset, operations).expand_dims(
                    dim={"feature": [gdf.iloc[gdf_idx].name]}
                )
            )

        del yx_pos, feats, shapes

    else:
        for idx_gdb, feat in tqdm.tqdm(
            gdf.iterrows(), total=gdf.shape[0], disable=not verbose
        ):
            if feat.geometry.geom_type == "MultiPolygon":
                shapes = feat.geometry.geoms
            else:
                shapes = [feat.geometry]
            datacube_spatial_subset = dataset.rio.clip(shapes, all_touched=all_touched)

            zonal_feat = _zonal_stats_geom(
                datacube_spatial_subset, operations
            ).expand_dims(dim={"feature": [feat.name]})

            zonal_ds_list.append(zonal_feat)

    return xr.concat(zonal_ds_list, dim="feature")
