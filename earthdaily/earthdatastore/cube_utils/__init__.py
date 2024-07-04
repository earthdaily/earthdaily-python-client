from collections import defaultdict
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from shapely.geometry import box
from .geometry_manager import GeometryManager
from ._zonal import zonal_stats, zonal_stats_numpy
from .harmonizer import Harmonizer
from .asset_mapper import AssetMapper
import rioxarray
from functools import wraps
import json

__all__ = ["GeometryManager", "rioxarray", "zonal_stats", "zonal_stats_numpy"]


def _datacube_masks(method, *args, **kwargs):
    @wraps(method)
    def _impl(self, *args, **kwargs):
        mask_with = kwargs.get("mask_with", None)
        if isinstance(mask_with, list):
            kwargs.pop("mask_with")
            for idx, mask in enumerate(mask_with):
                try:
                    datacube = method(self, mask_with=mask, *args, **kwargs)
                    break
                except Warning as E:
                    # if warning about no items for ag_cloud_mask for example
                    if idx + 1 == len(mask_with):
                        raise E
        else:
            datacube = method(self, *args, **kwargs)
        return datacube

    return _impl


def _datacubes(method):
    @wraps(method)
    @_datacube_masks
    def _impl(self, *args, **kwargs):
        collections = kwargs.get("collections", args[0] if len(args) > 0 else None)
        if isinstance(collections, list) and len(collections) > 1:
            if "collections" in kwargs.keys():
                kwargs.pop("collections")
            else:
                args = args[1:]
            datacubes = []
            for idx, collection in enumerate(collections):
                datacube = method(self, collections=collection, *args, **kwargs)
                if idx == 0:
                    kwargs["geobox"] = datacube.odc.geobox
                datacubes.append(datacube)
            datacube = metacube(*datacubes)
        else:
            datacube = method(self, *args, **kwargs)
        return datacube

    return _impl


def _match_xy_dims(src, dst, resampling=Resampling.nearest):
    if (src.sizes["x"], src.sizes["y"]) != (dst.sizes["x"], dst.sizes["y"]):
        src = src.rio.reproject_match(dst, resampling=resampling)
    return src


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
    attrs = {c: ds.coords[c].data.tolist() for c in ds.coords if c not in ds.sizes}
    # force str
    for attr in attrs:
        if not isinstance(attrs[attr], (str, int, float, np.ndarray, list, tuple)):
            ds.coords[attr] = str(attrs[attr])
            ds.coords[attr] = ds.coords[attr].astype(str)
    return ds


def _cube_odc(
    items_collection,
    assets=None,
    times=None,
    dtype="float32",
    properties=False,
    **kwargs,
):
    from odc import stac

    if "epsg" in kwargs:
        kwargs["crs"] = f"EPSG:{kwargs['epsg']}"
        kwargs.pop("epsg")
    if "resampling" in kwargs:
        if isinstance(kwargs["resampling"], int):
            kwargs["resampling"] = Resampling(kwargs["resampling"]).name

    kwargs["chunks"] = kwargs.get("chunks", dict(x="auto", y="auto", time=1))

    if "geobox" in kwargs.keys() and "geopolygon" in kwargs.keys():
        kwargs.pop("geopolygon")

    ds = stac.load(
        items_collection,
        bands=assets,
        preserve_original_order=True,
        dtype=dtype,
        groupby=None,
        **kwargs,
    )
    if properties:
        metadata = defaultdict(list)
        for i in items_collection:
            # if properties is only a key
            if isinstance(properties, str):
                metadata[properties].append(i.properties[properties])
            else:
                for k, v in i.properties.items():
                    if isinstance(properties, list):
                        if k not in properties:
                            continue
                    if isinstance(v, list):
                        v = str(v)
                    metadata[k].append(v)
        # to avoid mismatch if some properties are not available on all items
        df = pd.DataFrame.from_dict(metadata, orient="index").T
        # convert to xarray needs
        metadata = {k: ("time", v.tolist()) for k, v in df.items()}
        # assign metadata as coords
        ds = ds.assign_coords(**metadata)
    ds = ds.chunk(kwargs["chunks"])

    return ds


def _cube_stackstac(items_collection, assets=None, times=None, **kwargs):
    from stackstac import stack

    if "epsg" in kwargs:
        kwargs["epsg"] = int(kwargs["epsg"])
    if kwargs.get("geobox") is not None:
        kwargs["resolution"] = kwargs["geobox"].resolution.x
        kwargs["epsg"] = kwargs["geobox"].crs.to_epsg()
    if "geobox" in kwargs.keys():
        kwargs.pop("geobox")

    ds = stack(
        items_collection,
        assets=assets,
        rescale=False,
        xy_coords="center",
        **kwargs,
    )
    ds = ds.to_dataset(dim="band")
    if "band" in ds.sizes:
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
    common_band_names=True,
    cross_calibration_items: list | None = None,
    properties: (bool | str | list) = False,
    **kwargs,
):
    logging.info(f"Building datacube with {len(items_collection)} items")
    times = [
        np.datetime64(d.datetime.strftime("%Y-%m-%d %X.%f")).astype("datetime64[ns]")
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
    if common_band_names and not isinstance(assets, dict):
        aM = AssetMapper()
        assets = aM.map_collection_assets(items_collection[0].collection_id, assets)
    if isinstance(assets, dict):
        assets_keys = list(assets.keys())
    if engine == "odc" and intersects is not None:
        kwargs["geopolygon"] = GeometryManager(intersects).to_geopandas()
    if engine == "stackstac" and intersects is not None:
        kwargs["bounds_latlon"] = list(
            GeometryManager(intersects).to_geopandas().to_crs(epsg=4326).total_bounds
        )

    ds = engines[engine](
        items_collection,
        assets=assets_keys if isinstance(assets, dict) else assets,
        times=times,
        properties=properties,
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
    if groupby_date:
        if ds.time.size != np.unique(ds.time).size:
            ds = ds.groupby("time")
            ds = getattr(ds, groupby_date)()
            # get grouped value if several tiles at same exactly date
    if bbox is not None and intersects is None:
        intersects = _bbox_to_intersects(bbox)
    if intersects is not None:
        intersects = GeometryManager(intersects).to_geopandas()

    if isinstance(intersects, gpd.GeoDataFrame):
        # optimize by perclipping using bbox
        # no need anymore thanks to geobox/geopolygon in doc
        # ds = ds.rio.clip_box(*intersects.to_crs(ds.rio.crs).total_bounds)
        ds = ds.rio.clip(intersects.to_crs(ds.rio.crs).geometry)
    # apply nodata
    ds = _apply_nodata(ds, nodatas)
    if rescale:
        ds = rescale_assets_with_items(items_collection, ds, assets=assets)
    if engine == "stackstac":
        ds = _autofix_unfrozen_coords_dtype(ds)
    if cross_calibration_items is not None and len(cross_calibration_items) > 0:
        ds = Harmonizer.harmonize(items_collection, ds, cross_calibration_items, assets)
    if groupby_date:
        if ds.time.size != np.unique(ds.time.dt.strftime("%Y%m%d")).size:
            ds = ds.groupby("time.date")
            ds = getattr(ds, groupby_date)().rename(dict(date="time"))
    ds["time"] = ds.time.astype("<M8[ns]")

    if isinstance(assets, dict):
        ds = ds.rename(assets)

    for coord in ds.coords:
        if ds.coords[coord].values.shape == ():
            continue
        if isinstance(ds.coords[coord].values[0], (list, dict)):
            ds.coords[coord].values = [
                json.dumps(ds.coords[coord].values[idx])
                for idx in range(ds.coords[coord].size)
            ]

    return ds


def rescale_assets_with_items(
    items_collection,
    ds,
    assets=None,
    boa_offset_applied_control=True,
    boa_offset_applied_force_by_date=True,
):
    logging.info("rescale dataset")
    scales = dict()
    if len(items_collection) > ds.time.size:
        unique_dt = {}
        # items_collection_unique_dt = []
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
            if boa_offset_applied_force_by_date:
                yyyymmdd = np.datetime_as_string(time)[:10].replace("-", "")
                if yyyymmdd >= "20220228":
                    boa_offset_applied = True

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
                continue
            offset = rasterbands.get("offset", None)
            scale = rasterbands.get("scale", None)

            if offset or scale:
                if ds_asset not in scales:
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
                            "nir",
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
                    times = np.in1d(ds.time, list(set(scales[asset][scale][offset])))
                    asset_scaled = ds[[asset]].isel(time=times) * scale + offset
                    ds_scaled[asset].append(asset_scaled)
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
    unfrozen_coords = [i for i in list(ds.coords) if i not in ds.sizes]
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
    x_size = list(set(cube.sizes["x"] for cube in cubes))
    y_size = list(set(cube.sizes["y"] for cube in cubes))
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
    cube = cube.sortby(cube.time)
    return _propagade_rio(cubes[0], cube)
