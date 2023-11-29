import logging
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from shapely.geometry import box

from .asset_mapper import AssetMapper
from .geometry_manager import GeometryManager

__all__ = ["GeometryManager", "rioxarray"]


def _match_xy_dims(src, dst):
    if src.dims != dst.dims:
        src = src.rio.reproject_match(dst)
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
    attrs = {c: ds.coords[c].data.tolist() for c in ds.coords if c not in ds.dims}
    # force str
    for attr in attrs:
        if not isinstance(attrs[attr], (str, int, float, np.ndarray, list, tuple)):
            ds.coords[attr] = str(attrs[attr])
            ds.coords[attr] = ds.coords[attr].astype(str)
    return ds


def _cube_odc(items_collection, assets=None, times=None, dtype="float32", **kwargs):
    from odc import stac

    if "epsg" in kwargs:
        kwargs["crs"] = f"EPSG:{kwargs['epsg']}"
        kwargs.pop("epsg")
    if "resampling" in kwargs:
        if isinstance(kwargs["resampling"], int):
            kwargs["resampling"] = Resampling(kwargs["resampling"]).name
    kwargs["chunks"] = kwargs.get("chunks", dict(x=512, y=512, time=1))

    ds = stac.load(
        items_collection,
        bands=assets,
        preserve_original_order=True,
        dtype=dtype,
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
    common_band_names=True,
    cross_cal_items: list | None = None,
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
    if common_band_names and not isinstance(assets, dict):
        aM = AssetMapper()
        assets = aM.map_collection_assets(items_collection[0].collection_id, assets)
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
        # itrscts = intersects.to_crs(ds.rio.crs).iloc[[0]]
        # optimize by perclipping using bbox
        ds = ds.rio.clip_box(*intersects.to_crs(ds.rio.crs).total_bounds)
        ds = ds.rio.clip(intersects.to_crs(ds.rio.crs).geometry)
    # apply nodata
    ds = _apply_nodata(ds, nodatas)
    if rescale:
        ds = rescale_assets_with_items(items_collection, ds, assets=assets)
    if engine == "stackstac":
        ds = _autofix_unfrozen_coords_dtype(ds)
    if cross_cal_items is not None and len(cross_cal_items) > 0:
        ds = apply_cross_calibration(items_collection, ds, cross_cal_items, assets)
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


def apply_cross_calibration(items_collection, ds, cross_cal_items, assets):
    if assets is None:
        assets = list(ds.data_vars.keys())

    scaled_dataset = {}

    # Initializing asset list
    for asset in assets:
        scaled_dataset[asset] = []

    # For each item in the datacube
    for idx, time in enumerate(ds.time.values):
        print("-----")
        current_item = items_collection[idx]
        print(current_item)

        # Filtering cross cal items
        # Check si on a item avec la platform de l'item
        # Si oui => on utilise ca
        # Si non =>
        # Check si item avec platform a "" (équivalent a coef pour tous les items)
        # Si oui => on utilise ca
        # Si non => on enleve l'image (pas de cross cal dispo)

        platform = current_item.properties["platform"]

        print(platform)

        # Looking for platform/camera specific xcal coef
        platform_xcal_items = [
            item
            for item in cross_cal_items
            if item.properties["eda_cross_cal:source_platform"] == platform
            and check_timerange(item, current_item.datetime)
        ]

        # at least one match
        matching_xcal_item = None
        if len(platform_xcal_items) > 0:
            matching_xcal_item = platform_xcal_items[0]
        else:
            # Looking for global xcal coef
            global_xcal_items = [
                item
                for item in cross_cal_items
                if item.properties["eda_cross_cal:source_platform"] == ""
                and check_timerange(item, current_item.datetime)
            ]

            if len(global_xcal_items) > 0:
                matching_xcal_item = cross_cal_items[0]

        if matching_xcal_item is not None:
            print("XCAL COEF FOUND")
            print(matching_xcal_item.id)

            for ds_asset in assets:
                # Loading Xcal coef for the specific band
                asset_xcal_coef = matching_xcal_item.properties["eda_cross_cal:bands"][
                    ds_asset
                ]

                if asset_xcal_coef:
                    # Apply XCAL
                    print("HERE WE APPLY THE CROSS CAL FOR " + ds_asset)
                    # By default, we take the first item we have
                    scaled_asset = apply_cross_calibration_to_asset(
                        asset_xcal_coef[0][ds_asset],
                        ds[[ds_asset]].loc[dict(time=time)],
                        ds_asset,
                    )
                    scaled_dataset[ds_asset].append(scaled_asset)
        else:
            print("XCAL COEF NOT FOUND")

    ds_ = []
    for k, v in scaled_dataset.items():
        ds_k = []
        for d in v:
            ds_k.append(d)
        ds_.append(xr.concat(ds_k, dim="time"))
    ds_ = xr.merge(ds_).sortby("time")
    ds_.attrs = ds.attrs

    return ds_


def check_timerange(xcal_item, item_datetime):
    start_date = datetime.strptime(
        xcal_item.properties["published"], "%Y-%m-%dT%H:%M:%SZ"
    )
    start_date = start_date.replace(tzinfo=pytz.UTC)

    # print("----------------")
    # print(f"Item date : {item_datetime}")
    # print(f"Range Start date : {start_date}")

    if "expires" in xcal_item.properties:
        end_date = datetime.strptime(
            xcal_item.properties["expires"], "%Y-%m-%dT%H:%M:%SZ"
        )
        end_date = end_date.replace(tzinfo=pytz.UTC)

        # print(f"Range End date : {end_date}")
        # print(start_date <= item_datetime <= end_date)

        return start_date <= item_datetime <= end_date
    else:
        # print(start_date <= item_datetime)
        return start_date <= item_datetime


def define_functions_from_xcal_item(functions):
    possible_range = ["ge", "gt", "le", "lt"]

    operator_mapping = {
        "lt": "<",
        "le": "<=",
        "ge": ">=",
        "gt": ">",
    }

    xarray_where_concat = ""
    for idx_function, function in enumerate(functions):
        coef_range = [function["range_start"], function["range_end"]]

        for idx_coef, coef_border in enumerate(coef_range):
            for range in possible_range:
                if range in coef_border:
                    threshold = coef_border[range]
                    condition = operator_mapping.get(range)
                    if idx_coef == 0:
                        xarray_where_concat += f"xr.where((x{condition}{threshold})"
                    else:
                        xarray_where_concat += f" & (x{condition}{threshold})"

        xarray_where_concat += f',x * {function["scale"]} + {function["offset"]},'

        if idx_function == len(functions) - 1:
            xarray_where_concat += "x"
            function_parenthesis = 0

            while function_parenthesis < len(functions):
                xarray_where_concat += ")"
                function_parenthesis += 1

    # print(xarray_where_concat)

    return xarray_where_concat


def apply_cross_calibration_to_asset(functions, asset, band_name):
    if len(functions) == 1:
        # Single function
        return asset * functions[0]["scale"] + functions[0]["offset"]
    else:
        # Multiple functions
        x = asset[band_name]
        xr_where_string = define_functions_from_xcal_item(functions)
        asset[band_name] = eval(xr_where_string)
        return asset
