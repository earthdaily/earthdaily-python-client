import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rasterio.enums import Resampling
from rasterio.mask import geometry_mask
import xarray as xr
import rioxarray as rxr
import tqdm

def _xrmode(arr, **kwargs):
    values, counts = np.unique(arr, return_counts=True)
    isnan = np.isnan(values)
    values, counts = values[~isnan], counts[~isnan]
    return values[np.argmax(counts)]


def zonal_stats(
    dataset,
    gdf,
    operations=["mean"],
    all_touched=False,
    preload_time=True,
):
    """


    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with time dimension.
    gdf : gpd.GeoDataFrame
        dataframe index will be store in "feature_name" dimension.
    operations : list, optional
        List of mathematical operations you want to apply. The default is ["mean"].
        Available : ["count", "mean", "median", "max", "min", "std", "mode"]

    all_touched : bool, optional
        Include a pixel in the mask if it touches any of the shapes. If False (default), include a pixel only if its center is within one of the shapes, or if it is selected by Bresenham’s line algorithm.
        The default is False.
    preload_time : bool, optional
        Load each time dataset in memory to do not load it for each geometry.
        The default is True.

    Raises
    ------
    NotImplementedError
        Only supports xr.Dataset.

    Returns
    -------
    xr.Dataset
        3 dimensions (feature_name, stats, time)

            feature_name : Index value of the input feature.

            stats : list of chosen operations.

            time : same as the input time dimension

    """
    if not isinstance(dataset, xr.Dataset):
        raise NotImplementedError("Only support xarray dataset for the moment.")

    valid_operations = ["count", "mean", "median", "max", "min", "std", "mode"]
    for operation in operations:
        if operation not in valid_operations:
            raise NotImplementedError(
                f"{operation} is not an operation available. Please use on the these : {valid_operations}"
            )

    def feature_stat(idx_gdb, feat, dataset, operations, all_touched=all_touched):
        global ds_stats
        if feat.geometry.geom_type == "MultiPolygon":
            shapes = feat.geometry.geoms
        else:
            shapes = [feat.geometry]
        subset_shape = dataset.rio.clip(shapes, all_touched=all_touched)

        for idx, varname in enumerate(dataset.data_vars):
            subset = subset_shape[varname]
            stats = {}
            for operation in operations:
                if operation in valid_operations:
                    if operation == "mode":
                        stats[operation] = subset.groupby("time").reduce(
                            _xrmode,
                            list(dim for dim in subset.dims if dim != "time"),
                        )
                    else:
                        operation_func = getattr(subset.groupby("time"), operation)
                        stats[operation] = operation_func(...)
                    stats[operation].expand_dims(dim={"stats": [operation]})

            ds_stats_ = xr.merge(list(stats.values()))
            ds_stats_ = ds_stats_.expand_dims(dim={"feature_name": [feat.name]})
            if idx == 0:
                ds_stats = ds_stats_.copy()
            else:
                ds_stats = xr.merge((ds_stats, ds_stats_))
        return ds_stats

    if gdf.crs != dataset.rio.crs:
        Warning(
            f"Different projections. Reproject vector to EPSG:{dataset.rio.crs.to_epsg()}."
        )
    gdf = gdf.to_crs(dataset.rio.crs)

    zonal_ds = []
    for time_idx in tqdm.tqdm(range(dataset.time.size), total=dataset.time.size):
        for idx_gdb, feat in gdf.iterrows():
            zonal_feat = feature_stat(
                idx_gdb,
                feat,
                dataset.isel(time=[time_idx]).load()
                if preload_time
                else dataset.isel(time=[time_idx]),
                operations,
            )
            zonal_ds.append(zonal_feat)

    zonal_ds = xr.merge(zonal_ds)

    return zonal_ds


def _bbox_to_intersects(bbox):
    if isinstance(bbox, str):
        bbox = [float(i) for i in bbox.split(",")]
    return gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")


def _apply_nodata(ds, nodata_assets: dict):
    for asset, nodata in nodata_assets.items():
        ds[asset].rio.set_nodata(nodata)
        ds[asset] = ds[asset].where(ds[asset] != nodata)
    return ds


def _autofix_unfrozen_coords_dtype(ds):
    attrs = {c: ds.coords[c].data.tolist() for c in ds.coords if c not in ds.dims}
    # force str
    for attr in attrs:
        if not isinstance(attrs[attr], (str, int, float, np.ndarray, list, tuple)):
            ds.coords[attr] = str(attrs[attr])
            ds.coords[attr] = ds.coords[attr].astype(str)
    return ds


def _cube_odc(items_collection, assets=None, times=None, **kwargs):
    from odc import stac

    if "epsg" in kwargs:
        kwargs["crs"] = f"EPSG:{kwargs['epsg']}"
        kwargs.pop("epsg")
    if "resampling" in kwargs:
        if isinstance(kwargs["resampling"], int):
            kwargs["resampling"] = Resampling(kwargs["resampling"]).name
    chunks = kwargs.get("chunks", dict(x=2048, y=2048, time=1))
    kwargs.pop("chunks", None)

    ds = stac.load(
        items_collection,
        bands=assets,
        chunks=chunks,
        preserve_original_order=True,
        dtype="float64",
        groupby=None,
        **kwargs,
    )

    return ds


def _cube_stackstac(items_collection, assets=None, times=None, **kwargs):
    from stackstac import stack

    if "epsg" in kwargs:
        kwargs["epsg"] = int(kwargs["epsg"])
    ds = stack(
        items_collection,
        assets=assets,
        rescale=False,
        xy_coords="center",
        properties=True,
        **kwargs,
    )
    ds = ds.to_dataset(dim="band")
    if "band" in ds.dims:
        ds = ds.drop_dims("band")
    for data_vars in ds.data_vars:
        ds[data_vars] = ds[data_vars].rio.write_crs(ds.rio.crs)
    if times:
        ds["time"] = times

    return ds


def datacube(
    items_collection=None,
    bbox=None,
    intersects=None,
    assets: list | dict = None,
    engine="odc",
    rescale=True,
    groupby_date="mean",
    **kwargs,
):
    logging.info(f"Building datacube with {len(items_collection)} items")
    times = [
        np.datetime64(d.datetime).astype("datetime64[ns]")
        for d in items_collection
        if "datetime" in d.__dict__
    ]

    if len(times) == 0:
        times = None
    engines = {"odc": _cube_odc, "stackstac": _cube_stackstac}
    if engine not in engines:
        raise NotImplementedError(
            f"Engine '{engine}' not supported. Only {' and '.join(list(engines.keys()))} are currently supported."
        )
    if isinstance(assets, dict):
        assets_keys = list(assets.keys())
    ds = engines[engine](
        items_collection,
        assets=assets_keys if isinstance(assets, dict) else assets,
        times=times,
        **kwargs,
    )
    nodatas = {}
    for ds_asset in ds.data_vars:
        for item in items_collection:
            empty_dict_list = []
            band_idx = 1
            asset = ds_asset
            if len(ds_asset.split(".")) == 2:
                asset, band_idx = asset.split(".")
                band_idx = int(band_idx)
            for i in range(band_idx + 1):
                empty_dict_list.append({})
            if asset not in item.assets.keys():
                continue
            nodata = (
                item.assets[asset]
                .extra_fields.get("raster:bands", empty_dict_list)[band_idx - 1]
                .get("nodata")
            )
            if nodata == 0 or nodata:
                nodatas.update({ds_asset: nodata})

    # drop na dates
    ds = ds.isel(dict(time=np.where(~np.isnan(ds.time))[0]))
    if ds.time.size != np.unique(ds.time).size:
        # get grouped value if several tiles at same exactly date
        if groupby_date:
            ds = ds.groupby("time")
            ds = getattr(ds, groupby_date)()
        else:
            ds = ds.groupby("time").mean()
    if bbox and intersects is None:
        intersects = _bbox_to_intersects(bbox)
    if isinstance(intersects, gpd.GeoDataFrame):
        itrscts = intersects.to_crs(ds.rio.crs).iloc[[0]]
        # optimize by perclipping using bbox
        ds = ds.sel(
            x=slice(itrscts.bounds.minx[0], itrscts.bounds.maxx[0]),
            y=slice(itrscts.bounds.maxy[0], itrscts.bounds.miny[0]),
        )
        # do the perfect clip :)
        clip_mask_arr = geometry_mask(
            geometries=itrscts.geometry,
            out_shape=(int(ds.rio.height), int(ds.rio.width)),
            transform=ds.rio.transform(recalc=True),
            all_touched=False,
            invert=True,
        )

        mask_ = xr.DataArray(data=clip_mask_arr, coords=dict(y=ds.y, x=ds.x)).chunk(
            chunks=dict(x=ds.chunks["x"][0], y=ds.chunks["y"][0])
        )

        ds = ds.where(mask_)
        del mask_, clip_mask_arr
    # apply nodata
    ds = _apply_nodata(ds, nodatas)
    if rescale:
        ds = rescale_assets_with_items(items_collection, ds, assets=assets)
    if engine == "stackstac":
        ds = _autofix_unfrozen_coords_dtype(ds)
    if groupby_date:
        if ds.time.size != np.unique(ds.time.dt.strftime("%Y%m%d")).size:
            ds = ds.groupby("time.date")
            ds = getattr(ds, groupby_date)().rename(dict(date="time"))
    ds["time"] = ds.time.astype("<M8[ns]")

    if isinstance(assets, dict):
        ds = ds.rename(assets)
    return ds


def rescale_assets_with_items(
    items_collection, ds, assets=None, boa_offset_applied_control=True
):
    logging.info("rescale dataset")
    scales = dict()
    if len(items_collection) > ds.time.size:
        unique_dt = {}
        items_collection_unique_dt = []
        for item in items_collection:
            if item.datetime in unique_dt.keys():
                for asset in item.assets.keys():
                    if asset not in unique_dt[item.datetime].assets.keys():
                        unique_dt[item.datetime].assets[asset] = item.assets[asset]
            else:
                unique_dt[item.datetime] = item
            # items_collection_unique_dt.append(item)
        items_collection = [i for i in unique_dt.values()]
        if len(items_collection) != ds.time.size:
            raise ValueError(
                "Mismatch between number of items and datacube time, so cannot scale data. Please set rescale to False."
            )

    for idx, time in enumerate(ds.time.values):
        current_item = items_collection[idx]
        if pd.Timestamp(time).strftime("%Y%m%d") != current_item.datetime.strftime(
            "%Y%m%d"
        ):
            raise ValueError(
                "Mismatch between items and datacube, cannot scale data. Please set rescale to False."
            )
        if (
            current_item.collection_id == "sentinel-2-l2a"
            and boa_offset_applied_control
        ):
            boa_offset_applied = items_collection[idx].properties.get(
                "earthsearch:boa_offset_applied", False
            )
        if assets is None:
            assets = list(ds.data_vars.keys())
        for ds_asset in assets:
            empty_dict_list = []
            band_idx = 1
            asset = ds_asset
            if len(ds_asset.split(".")) == 2:
                asset, band_idx = asset.split(".")
                band_idx = int(band_idx)
            for i in range(band_idx + 1):
                empty_dict_list.append(False)
            if asset not in current_item.assets.keys():
                continue
            rasterbands = current_item.assets[asset].extra_fields.get(
                "raster:bands", empty_dict_list
            )[band_idx - 1]

            if rasterbands is False:
                break
            offset = rasterbands.get("offset", None)
            scale = rasterbands.get("scale", None)

            if offset or scale:
                if not ds_asset in scales:
                    scales[ds_asset] = {}

                scale = rasterbands.get("scale", 1)
                offset = rasterbands.get("offset", 0)
                if (
                    current_item.collection_id == "sentinel-2-l2a"
                    and boa_offset_applied_control
                ):
                    if (
                        ds_asset
                        in [
                            "blue",
                            "red",
                            "green",
                            "nir08",
                            "nir09",
                            "swir16",
                            "swir22",
                            "rededge1",
                            "rededge2",
                            "rededge3",
                        ]
                        and boa_offset_applied
                    ):
                        offset = 0
                if scales[ds_asset].get(scale, None) is None:
                    scales[ds_asset][scale] = {}
                if scales[ds_asset][scale].get(offset, None) is None:
                    scales[ds_asset][scale][offset] = []
                scales[ds_asset][scale][offset].append(time)
    if len(scales) > 0:
        ds_scaled = {}
        for asset in scales.keys():
            ds_scaled[asset] = []
            for scale in scales[asset].keys():
                for offset in scales[asset][scale].keys():
                    times = list(set(scales[asset][scale][offset]))
                    if len(times) != len(scales[asset][scale][offset]):
                        for time in times:
                            d = ds[[asset]].loc[dict(time=time)] * scale + offset
                            ds_scaled[asset].append(d)
                    else:
                        d = ds[[asset]].loc[dict(time=times)] * scale + offset
                        ds_scaled[asset].append(d)
        ds_ = []
        for k, v in ds_scaled.items():
            ds_k = []
            for d in v:
                ds_k.append(d)
            ds_.append(xr.concat(ds_k, dim="time"))
        attrs = ds.attrs
        ds_ = xr.merge(ds_).sortby("time")
        missing_vars = [d for d in ds.data_vars if d not in scales.keys()]
        if len(missing_vars) > 0:
            ds_ = xr.merge([ds_, ds[missing_vars]])
        ds_.attrs = attrs
        ds = ds_
        del ds_scaled, scales
    logging.info("rescale done")
    ds = ds.sortby("time")
    return ds


def _propagade_rio(src, ds):
    coords = ["epsg", "spatial_ref"]
    for coord in coords:
        if coord in src:
            ds = ds.assign_coords(coord=src[coord])
    return ds


def _drop_unfrozen_coords(ds):
    unfrozen_coords = [i for i in list(ds.coords) if i not in ds.dims]
    ds = ds.drop(unfrozen_coords)
    return ds


def _common_data_vars(*cubes):
    data_vars = list(set([k for cube in cubes for k in list(cube.data_vars.keys())]))
    return data_vars


def _groupby(ds, by="time.date", how="mean"):
    condition = getattr(ds, by).size != np.unique(getattr(ds, by)).size
    if condition:
        ds = ds.groupby(by)
        ds = getattr(ds, how)()
        if by == "time.date":
            ds = ds.rename(dict(date="time"))
            ds["time"] = ds.time.astype("<M8[ns]")
    return ds


def _have_same_xy(*cubes):
    x_size = list(set(cube.dims["x"] for cube in cubes))
    y_size = list(set(cube.dims["y"] for cube in cubes))
    if len(x_size) == 1 and len(y_size) == 1:
        return True
    return False


def metacube(*cubes, concat_dim="time", by="time.date", how="mean"):
    if not _have_same_xy(*cubes):
        raise ValueError("Cubes must have same x and y dimensions.")
    data_vars = _common_data_vars(*cubes)
    for idx, cube in enumerate(cubes):
        first_available_var = list(cube.data_vars.keys())[0]
        for data_var in data_vars:
            if data_var not in cube.data_vars:
                cubes[idx][data_var] = cubes[idx][first_available_var]
                # fill with nan
                cubes[idx][data_var] = cubes[idx][data_var].where(
                    cubes[idx][data_var] == np.nan, other=np.nan
                )

    cube = xr.concat([_drop_unfrozen_coords(cube) for cube in cubes], dim=concat_dim)
    cube = _groupby(cube, by=by, how=how)
    return _propagade_rio(cubes[0], cube)
