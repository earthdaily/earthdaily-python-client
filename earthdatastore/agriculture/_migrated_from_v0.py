import json
import logging
import operator
import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import pytz
import requests
import shapely
import toml
import tqdm
import xarray as xr
from joblib import Parallel, delayed
from odc import stac
from pandas import Timedelta, Timestamp
from pystac.item_collection import ItemCollection
from pystac_client import Client
from pystac_client.item_search import ItemSearch
from pystac_client.stac_api_io import StacApiIO
from rasterio import features
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from scipy.sparse import csr_matrix
from scipy.stats import mode
from shapely.geometry import box
from stackstac import stack
from urllib3 import Retry

T = TypeVar("T")

DatetimeRange = Tuple[Union[datetime, Timestamp], Union[datetime, Timestamp]]
DateRangeList = List[DatetimeRange]

scale_factor_collections = {
    "landsat-c2l2-sr": [
        dict(
            assets=[
                "red",
                "blue",
                "green",
                "nir",
                "nir08",
                "swir16",
                "swir22",
                "coastal",
            ],
            scale=0.0000275,
            offset=-0.2,
            nodata=0,
        )
    ],
    "landsat-c2l2-st": [
        dict(
            assets=["lwir", "lwir11", "lwir12"],
            scale=0.00341802,
            offset=149.0,
            nodata=0,
        )
    ],
    "landsat-c2l1": [
        dict(
            assets=[
                "red",
                "blue",
                "green",
                "nir",
                "nir08",
                "swir16",
                "swir22",
                "coastal",
            ],
            scale=0.0001,
            offset=0,
            nodata=-9999,
        ),
        dict(
            assets=[
                "lwir",
                "lwir11",
                "lwir12",
            ],
            scale=0.1,
            offset=0,
            nodata=-9999,
        ),
    ],
}

_auto_mask_order = ["cloudmask", "ag_cloud_mask", "native"]


__pathFile = os.path.dirname(os.path.realpath(__file__))
__asset_mapper_config_path = os.path.join(__pathFile, "asset_mapper_config.json")
_asset_mapper_config = json.load(open(__asset_mapper_config_path))

dask.config.set(**{"array.slicing.split_large_chunks": True})

_available_masks = [
    "native",
    "venus_detailed_cloud_mask",
    "ag_cloud_mask",
    "cloud_mask",
    "ag-cloud-mask",
    "cloud_mask_ag_version",
    "cloudmask_ag_version",
    "cloudmask",
    "scl",
]
_native_mask_def_mapping = {
    "sentinel-2-l2a": "scl",
    "sentinel-2-c1-l2a": "scl",
    "venus-l2a": "venus_detailed_cloud_mask",
    "landsat-c2l2-sr": "landsat_qa_pixel",
    "landsat-c2l2-st": "landsat_qa_pixel",
}
_native_mask_asset_mapping = {
    "sentinel-2-l2a": "scl",
    "sentinel-2-c1-l2a": "scl",
    "venus-l2a": "detailed_cloud_mask",
    "landsat-c2l2-sr": "qa_pixel",
    "landsat-c2l2-st": "qa_pixel",
}


@dataclass
class EarthDataStoreConfig:
    auth_url: Optional[str] = None
    client_secret: Optional[str] = None
    client_id: Optional[str] = None
    eds_url: str = "https://api.earthdaily.com/platform/v1/stac"
    access_token: Optional[str] = None


class Harmonizer:
    def harmonize(items_collection, ds, cross_cal_items, assets):
        """
        Harmonize a dataset using cross_cal items from EarthDaily collection.

        Parameters
        ----------
        items_collection : TYPE
            DESCRIPTION.
        ds : TYPE
            DESCRIPTION.
        cross_cal_items : TYPE
            DESCRIPTION.
        assets : TYPE
            DESCRIPTION.

        Returns
        -------
        ds_ : TYPE
            DESCRIPTION.

        """
        if assets is None:
            assets = list(ds.data_vars.keys())

        scaled_dataset = {}

        # Initializing asset list
        for asset in assets:
            scaled_dataset[asset] = []

        # For each item in the datacube
        for idx, time_x in enumerate(ds.time.values):
            current_item = items_collection[idx]

            platform = current_item.properties["platform"]

            # Looking for platform/camera specific xcal coef
            platform_xcal_items = [
                item
                for item in cross_cal_items
                if item.properties["eda_cross_cal:source_platform"] == platform
                and Harmonizer.check_timerange(item, current_item.datetime)
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
                    and Harmonizer.check_timerange(item, current_item.datetime)
                ]

                if len(global_xcal_items) > 0:
                    matching_xcal_item = cross_cal_items[0]

            if matching_xcal_item is not None:
                for ds_asset in assets:
                    # Loading Xcal coef for the specific band
                    bands_coefs = matching_xcal_item.properties["eda_cross_cal:bands"]

                    if ds_asset in bands_coefs:
                        asset_xcal_coef = matching_xcal_item.properties["eda_cross_cal:bands"][ds_asset]
                        # By default, we take the first item we have
                        scaled_asset = Harmonizer.apply_to_asset(
                            asset_xcal_coef[0][ds_asset],
                            ds[ds_asset].loc[dict(time=time_x)],
                            ds_asset,
                        )
                        scaled_dataset[ds_asset].append(scaled_asset)
                    else:
                        scaled_dataset[ds_asset].append(ds[ds_asset].loc[dict(time=time_x)])

        ds_ = []
        for k, v in scaled_dataset.items():
            ds_k = []
            for d in v:
                ds_k.append(d)
            ds_.append(xr.concat(ds_k, dim="time"))
        ds_ = xr.merge(ds_).sortby("time")
        ds_.attrs = ds.attrs

        return ds_

    def xcal_functions_parser(functions, dataarray: xr.DataArray):
        xscaled_dataarray = []
        for idx_function, function in enumerate(functions):
            coef_range = [function["range_start"], function["range_end"]]
            for idx_coef, coef_border in enumerate(coef_range):
                for single_operator, threshold in coef_border.items():
                    xr_condition = getattr(operator, single_operator)(dataarray, threshold)
                    if idx_coef == 0:
                        ops = xr_condition
                    else:
                        ops = np.logical_and(ops, xr_condition)
            xscaled_dataarray.append(dict(condition=ops, scale=function["scale"], offset=function["offset"]))

        for op in xscaled_dataarray:
            dataarray = xr.where(op["condition"], dataarray * op["scale"] + op["offset"], dataarray)
        return dataarray

    def apply_to_asset(functions, dataarray: xr.DataArray, band_name):
        if len(functions) == 1:
            # Single function
            dataarray = dataarray * functions[0]["scale"] + functions[0]["offset"]
        else:
            # Multiple functions
            # TO DO : Replace x variable and the eval(xr_where_string) by a native function
            dataarray = Harmonizer.xcal_functions_parser(functions, dataarray)
        return dataarray

    def check_timerange(xcal_item, item_datetime):
        start_date = datetime.strptime(xcal_item.properties["published"], "%Y-%m-%dT%H:%M:%SZ")
        start_date = start_date.replace(tzinfo=pytz.UTC)

        if not isinstance(item_datetime, datetime):
            item_datetime = datetime.strptime(item_datetime, "%Y-%m-%dT%H:%M:%SZ")
        item_datetime = item_datetime.replace(tzinfo=pytz.UTC)

        if "expires" in xcal_item.properties:
            end_date = datetime.strptime(xcal_item.properties["expires"], "%Y-%m-%dT%H:%M:%SZ")
            end_date = end_date.replace(tzinfo=pytz.UTC)

            return start_date <= item_datetime <= end_date
        else:
            return start_date <= item_datetime


class Mask:
    def __init__(self, dataset: xr.Dataset, intersects=None, bbox=None):
        self._obj = dataset
        if bbox is not None and intersects is None:
            intersects = _bbox_to_intersects(bbox)
        if isinstance(intersects, gpd.GeoDataFrame):
            intersects = intersects.to_crs(self._obj.rio.crs)
        self.intersects = intersects
        self.compute_available_pixels()

    def cloud_mask(
        self,
        mask_statistics=False,
    ):
        self._obj["time"] = self._obj.time.dt.round("s")  # rm nano second
        #
        self._obj = self._obj.where(self._obj["cloud_mask"] == 1)
        if mask_statistics:
            self.compute_clear_coverage(
                self._obj["ag_cloud_mask"],
                "ag_cloud_mask",
                1,
                labels_are_clouds=False,
            )
        return self._obj

    def ag_cloud_mask(
        self,
        mask_statistics=False,
    ):
        self._obj["time"] = self._obj.time.dt.round("s")  # rm nano second
        #
        self._obj = self._obj.where(self._obj["ag_cloud_mask"] == 1)
        if mask_statistics:
            self.compute_clear_coverage(self._obj["ag_cloud_mask"], "ag_cloud_mask", 1, labels_are_clouds=False)
        return self._obj

    def cloudmask_from_asset(
        self,
        cloud_asset,
        labels,
        labels_are_clouds,
        mask_statistics=False,
        fill_value=np.nan,
    ):
        if cloud_asset not in self._obj.data_vars:
            raise ValueError(f"Asset '{cloud_asset}' needed to compute cloudmask.")
        else:
            cloud_layer = self._obj[cloud_asset].copy()
        _assets = [a for a in self._obj.data_vars if a != cloud_asset]

        if fill_value:
            if labels_are_clouds:
                self._obj = self._obj.where(~self._obj[cloud_asset].isin(labels), fill_value)
            else:
                self._obj = self._obj.where(self._obj[cloud_asset].isin(labels), fill_value)
        if mask_statistics:
            self.compute_clear_coverage(cloud_layer, cloud_asset, labels, labels_are_clouds=labels_are_clouds)
        return self._obj

    def scl(
        self,
        clouds_labels=[1, 3, 8, 9, 10, 11],
        mask_statistics=False,
    ):
        return self.cloudmask_from_asset(
            cloud_asset="scl",
            labels=clouds_labels,
            labels_are_clouds=True,
            mask_statistics=mask_statistics,
        )

    def venus_detailed_cloud_mask(self, mask_statistics=False):
        return self.cloudmask_from_asset(
            "detailed_cloud_mask",
            0,
            labels_are_clouds=False,
            mask_statistics=mask_statistics,
        )

    def compute_clear_coverage(self, cloudmask_array, cloudmask_name, labels, labels_are_clouds=True):
        if self._obj.attrs.get("usable_pixels", None) is None:
            self.compute_available_pixels()

        n_pixels_as_labels = cloudmask_array.isin(labels).sum(dim=("x", "y"))
        if labels_are_clouds:
            n_pixels_as_labels = self._obj.attrs["usable_pixels"] - n_pixels_as_labels

        self._obj = self._obj.assign_coords({"clear_pixels": ("time", n_pixels_as_labels.data)})

        self._obj = self._obj.assign_coords(
            {
                "clear_percent": (
                    "time",
                    np.multiply(
                        self._obj["clear_pixels"].data / self._obj.attrs["usable_pixels"],
                        100,
                    ).astype(np.int8),
                )
            }
        )
        self._obj["clear_pixels"] = self._obj["clear_pixels"].load()
        self._obj["clear_percent"] = self._obj["clear_percent"].load()

        return self._obj

    def compute_available_pixels(self):
        if self.intersects is None:
            raise ValueError("bbox or intersects must be defined for now to compute cloud statistics.")

        clip_mask_arr = geometry_mask(
            geometries=self.intersects.geometry,
            out_shape=(int(self._obj.rio.height), int(self._obj.rio.width)),
            transform=self._obj.rio.transform(recalc=True),
            all_touched=False,
        )
        self.clip_mask_arr = clip_mask_arr
        usable_pixels = np.sum(np.in1d(clip_mask_arr, False))
        self._obj.attrs["usable_pixels"] = usable_pixels
        return self._obj

    def landsat_qa_pixel(self, mask_statistics=False):
        self._landsat_qa_pixel_convert()
        return self.cloudmask_from_asset(
            "qa_pixel",
            1,
            labels_are_clouds=False,
            mask_statistics=mask_statistics,
        )

    def _landsat_qa_pixel_convert(self):
        # load all time series to fasten next step
        if psutil.virtual_memory().available > self._obj["qa_pixel"].nbytes:
            self._obj["qa_pixel"] = self._obj["qa_pixel"].load()
        for time_x in self._obj.time:
            data = self._obj["qa_pixel"].loc[dict(time=time_x)].load().data
            data_f = data.flatten()
            clm = QA_PIXEL_cloud_detection(data_f[~np.isnan(data_f)])
            clm = np.where(clm == 0, np.nan, clm)
            data_f[~np.isnan(data_f)] = clm
            data = data_f.reshape(*data.shape)
            self._obj["qa_pixel"].loc[dict(time=time_x)] = data


def _QA_PIXEL_cloud_detection(pixel):
    """
    return 1 if cloudfree, 0 is cloud pixel
    """

    px_bin = np.binary_repr(pixel)
    if len(px_bin) == 15:
        reversed_bin = "0" + px_bin[::-1]
    elif len(px_bin) < 15:
        reversed_bin = "0000000000000000"
    else:
        reversed_bin = px_bin[::-1]

    if reversed_bin[7] == "0":
        return 0
    else:
        return 1


def QA_PIXEL_cloud_detection(arr):
    """
    return 1 if cloudfree, 0 is cloud pixel
    """
    uniques = np.unique(arr).astype(np.int16)
    has_cloud = np.array([_QA_PIXEL_cloud_detection(i) for i in uniques])
    cloudfree = np.where(has_cloud == 1, uniques, 0)
    cloudfree_pixels = cloudfree[cloudfree != 0]
    cloudmask = np.isin(arr, cloudfree_pixels).astype(int)
    return cloudmask


def filter_clear_cover(dataset, clear_cover, coordinate="clear_percent"):
    return dataset.isel(time=dataset[coordinate] >= clear_cover)


class AssetMapper:
    def __init__(self):
        self.available_collections = list(_asset_mapper_config.keys())

    def collection_mapping(self, collection):
        if self._collection_exists(collection, raise_warning=True):
            return _asset_mapper_config[collection]

    def _collection_exists(self, collection, raise_warning=False):
        exists = True if collection in self.available_collections else False
        if raise_warning and not exists:
            raise NotImplementedError(f"Collection {collection} has not been implemented")
        return exists

    def collection_spectral_assets(self, collection):
        return self.collection_mapping(collection)

    def map_collection_assets(self, collection, assets):
        if isinstance(assets, (dict | None)):
            return assets
        if not self._collection_exists(collection):
            return assets

        # HANDLE LIST TO DICT CONVERSION
        if isinstance(assets, list):
            assets = {asset: asset for asset in assets}

        output_assets = {}

        config = self.collection_mapping(collection)

        # Try to map each asset
        for asset in assets:
            if asset in config[0]:
                output_assets[config[0][asset]] = asset
            # No asset found with specified key (common asset name)
            else:
                # Looking for asset matching the specified value (asset name)
                matching_assets = [key for key, value in config[0].items() if value == asset]

                if matching_assets:
                    output_assets[asset] = asset
        return output_assets


def _datacube_masks(method: Callable) -> Callable:
    """
    Decorator to handle automatic mask selection and application.

    This decorator provides a flexible way to apply masks to a datacube,
    with an 'auto' mode that tries multiple mask options.

    Parameters
    ----------
    method : Callable
        The method to be wrapped with mask application logic.

    Returns
    -------
    Callable
        A wrapped method with enhanced mask handling capabilities.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Handle mask selection
        mask_with = kwargs.get("mask_with", None)

        # If 'auto' is specified, use the predefined auto mask order
        if isinstance(mask_with, str) and mask_with == "auto":
            mask_with = _auto_mask_order

        # If mask_with is a list, try each mask sequentially
        if isinstance(mask_with, list):
            kwargs.pop("mask_with", None)

            for mask in mask_with:
                try:
                    datacube = method(self, mask_with=mask, *args, **kwargs)
                    return datacube
                except Exception as error:
                    # If this is the last mask, re-raise the exception
                    if mask == mask_with[-1]:
                        raise error

        # If no special mask handling is needed, call the method directly
        return method(self, *args, **kwargs)

    return wrapper


def _datacubes(method: Callable) -> Callable:
    """
    Decorator to handle multiple collections and create meta-datacubes.

    This decorator provides logic for processing multiple collections,
    allowing creation of meta-datacubes when multiple collections are provided.

    Parameters
    ----------
    method : Callable
        The method to be wrapped with multi-collection handling logic.

    Returns
    -------
    Callable
        A wrapped method with enhanced multi-collection processing capabilities.
    """

    @wraps(method)
    @_datacube_masks
    def wrapper(self, *args, **kwargs):
        # Determine collections from args or kwargs
        collections = kwargs.get("collections", args[0] if len(args) > 0 else None)

        # If multiple collections are provided, process them separately
        if isinstance(collections, list) and len(collections) > 1:
            # Remove collections from kwargs or args
            if "collections" in kwargs:
                kwargs.pop("collections")
            else:
                args = args[1:]

            # Process each collection
            datacubes = []
            for idx, collection in enumerate(collections):
                # Create datacube for each collection
                datacube = method(self, collections=collection, *args, **kwargs)

                # Use the first datacube's geobox for subsequent datacubes
                if idx == 0:
                    kwargs["geobox"] = datacube.odc.geobox

                datacubes.append(datacube)

            # Combine datacubes into a meta-datacube
            return metacube(*datacubes)

        # If only one collection, process normally
        return method(self, *args, **kwargs)

    return wrapper


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
    if "epsg" in kwargs:
        kwargs["crs"] = f"EPSG:{kwargs['epsg']}"
        kwargs.pop("epsg")
    if "resampling" in kwargs:
        if isinstance(kwargs["resampling"], int):
            kwargs["resampling"] = Resampling(kwargs["resampling"]).name

    kwargs["chunks"] = kwargs.get("chunks", dict(x=512, y=512, time=1))

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
    if "latitude" in ds.coords and "longitude" in ds.coords:
        ds = ds.rename({"latitude": "y", "longitude": "x"})
    ds = ds.chunk(kwargs["chunks"])

    return ds


def _cube_stackstac(items_collection, assets=None, times=None, **kwargs):
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
        kwargs["bounds_latlon"] = list(GeometryManager(intersects).to_geopandas().to_crs(epsg=4326).total_bounds)

    # create datacube using the defined engine (default is odc stac)
    ds = engines[engine](
        items_collection,
        assets=assets_keys if isinstance(assets, dict) else assets,
        times=times,
        properties=properties,
        **kwargs,
    )

    # check nodata per asset (data_vars)
    # TODO : replace the original no_data with a defined value
    # (like min float) because of rescale

    nodatas = {}
    for ds_asset in ds.data_vars:
        for item in items_collection:
            empty_dict_list = []
            band_idx = 1
            asset = ds_asset
            if len(parts := ds_asset.split(".")) == 2:
                index = parts[1][-1]
                is_band = isinstance(index, int) or (isinstance(index, str) and index.isdigit())
                if is_band:
                    asset, band_idx = asset.split(".")
                    band_idx = int(band_idx)
            for i in range(band_idx + 1):
                empty_dict_list.append({})
            if asset not in item.assets.keys():
                continue
            nodata = item.assets[asset].extra_fields.get("raster:bands", empty_dict_list)[band_idx - 1].get("nodata")
            if nodata == 0 or nodata:
                nodatas.update({ds_asset: nodata})
            break

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
            ds.coords[coord].values = [json.dumps(ds.coords[coord].values[idx]) for idx in range(ds.coords[coord].size)]

    return ds


def rescale_assets_with_items(
    items_collection: list,
    ds: xr.Dataset,
    assets: None | list[str] = None,
    boa_offset_applied_control: bool = True,
    boa_offset_applied_force_by_date: bool = True,
) -> xr.Dataset:
    """
    Rescale assets in a dataset based on collection items' metadata.

    Parameters
    ----------
    items_collection : List
        Collection of items containing asset scaling information.
    ds : xarray.Dataset
        Input dataset to be rescaled.
    assets : List[str], optional
        List of assets to rescale. If None, uses all dataset variables.
    boa_offset_applied_control : bool, default True
        Apply Bottom of Atmosphere (BOA) offset control for Sentinel-2 L2A data.
    boa_offset_applied_force_by_date : bool, default True
        Force BOA offset application for dates after 2022-02-28.

    Returns
    -------
    xarray.Dataset
        Rescaled dataset with applied offsets and scales.

    Raises
    ------
    ValueError
        If there's a mismatch between items and datacube time or dates.
    """
    logging.info("Rescaling dataset")

    # Deduplicate items by datetime
    unique_items = {}
    for item in items_collection:
        unique_items.setdefault(item.datetime, item)

    items_collection_per_date = list(unique_items.values())

    # Validate items match dataset time
    if len(items_collection_per_date) != ds.time.size and len(items_collection) != ds.time.size:
        raise ValueError("Mismatch between items and datacube time. Set rescale to False.")

    # Prepare assets list
    assets = assets or list(ds.data_vars.keys())
    scales: dict[str, dict[float, dict[float, list]]] = {}

    # Process scaling for each time step
    for idx, time_x in enumerate(ds.time.values):
        item = items_collection[idx]

        # Date validation
        if pd.Timestamp(time_x).strftime("%Y%m%d") != item.datetime.strftime("%Y%m%d"):
            raise ValueError("Mismatch between items and datacube dates. Set rescale to False.")

        # BOA offset handling for Sentinel-2 L2A
        boa_offset_applied = item.properties.get("earthsearch:boa_offset_applied", False)
        if boa_offset_applied_control and item.collection_id == "sentinel-2-l2a":
            if boa_offset_applied_force_by_date:
                boa_offset_applied = pd.Timestamp(time_x) >= pd.Timestamp("2022-02-28")

        # Process each asset
        for ds_asset in assets:
            # Handle multi-band assets
            asset, band_idx = _parse_asset_band(ds_asset)

            if asset not in item.assets:
                continue

            raster_bands = item.assets[asset].extra_fields.get("raster:bands", [])
            if not raster_bands or len(raster_bands) < band_idx:
                continue

            rasterbands = raster_bands[band_idx - 1]
            scale = rasterbands.get("scale", 1)
            offset = rasterbands.get("offset", 0)

            # Special handling for Sentinel-2 BOA offset
            if (
                item.collection_id == "sentinel-2-l2a"
                and boa_offset_applied_control
                and ds_asset
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

            # Track scaling parameters
            scales.setdefault(ds_asset, {}).setdefault(scale, {}).setdefault(offset, []).append(time)

    # Apply rescaling
    if scales:
        scaled_assets = []
        for asset, scale_data in scales.items():
            asset_scaled = []
            for scale, offset_data in scale_data.items():
                for offset, times in offset_data.items():
                    mask = np.in1d(ds.time, times)
                    asset_scaled.append(ds[[asset]].isel(time=mask) * scale + offset)
            scaled_assets.append(xr.concat(asset_scaled, dim="time"))

        # Merge scaled assets
        ds_scaled = xr.merge(scaled_assets).sortby("time")

        # Preserve unscaled variables
        missing_vars = [var for var in ds.data_vars if var not in scales]
        if missing_vars:
            ds_scaled = xr.merge([ds_scaled, ds[missing_vars]])

        ds_scaled.attrs = ds.attrs
        ds = ds_scaled

    logging.info("Rescaling complete")
    return ds.sortby("time")


def _parse_asset_band(ds_asset: str) -> tuple[str, int]:
    """
    Parse asset and band index from asset name.

    Parameters
    ----------
    ds_asset : str
        Asset name, potentially with band index.

    Returns
    -------
    tuple[str, int]
        Tuple of (asset name, band index)
    """
    parts = ds_asset.split(".")
    if len(parts) == 2 and parts[1][-1].isdigit():
        return parts[0], int(parts[1][-1])
    return ds_asset, 1


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
                cubes[idx][data_var] = cubes[idx][data_var].where(cubes[idx][data_var] == np.nan, other=np.nan)
    cube = xr.concat([_drop_unfrozen_coords(cube) for cube in cubes], dim=concat_dim)
    cube = _groupby(cube, by=by, how=how)
    cube = cube.sortby(cube.time)
    return _propagade_rio(cubes[0], cube)


class GeometryManager:
    """
    A class to manage and convert various types of geometries into different formats.

    Parameters
    ----------
    geometry : various types
        Input geometry, can be GeoDataFrame, GeoSeries, WKT string, or GeoJSON.

    Attributes
    ----------
    geometry : various types
        The input geometry provided by the user.
    _obj : GeoDataFrame
        The geometry converted into a GeoDataFrame.
    input_type : str
        Type of input geometry inferred during processing.

    Methods
    -------
    __call__():
        Returns the stored GeoDataFrame.
    to_intersects(crs='EPSG:4326'):
        Converts geometry to GeoJSON intersects format with a specified CRS.
    to_wkt(crs='EPSG:4326'):
        Converts geometry to WKT format with a specified CRS.
    to_json(crs='EPSG:4326'):
        Converts geometry to GeoJSON format with a specified CRS.
    to_geopandas():
        Returns the GeoDataFrame of the input geometry.
    to_bbox(crs='EPSG:4326'):
        Returns the bounding box of the geometry with a specified CRS.
    buffer_in_meter(distance, crs_meters='EPSG:3857', **kwargs):
        Applies a buffer in meters to the geometry and returns it with the original CRS.
    """

    def __init__(self, geometry):
        self.geometry = geometry
        self._obj = self.to_geopandas()

    def __call__(self):
        """Returns the GeoDataFrame stored in the object."""
        return self._obj

    def to_intersects(self, crs="EPSG:4326"):
        """
        Converts the geometry to GeoJSON intersects format.

        Parameters
        ----------
        crs : str, optional
            The coordinate reference system (CRS) to convert to (default is EPSG:4326).

        Returns
        -------
        dict
            The geometry in GeoJSON intersects format.
        """
        return json.loads(self._obj.to_crs(crs).dissolve().geometry.to_json())["features"][0]["geometry"]

    def to_wkt(self, crs="EPSG:4326"):
        """
        Converts the geometry to WKT format.

        Parameters
        ----------
        crs : str, optional
            The CRS to convert to (default is EPSG:4326).

        Returns
        -------
        str or list of str
            The geometry in WKT format. If there is only one geometry, a single WKT
            string is returned; otherwise, a list of WKT strings.
        """
        wkts = list(self._obj.to_crs(crs=crs).to_wkt()["geometry"])
        return wkts[0] if len(wkts) == 1 else wkts

    def to_json(self, crs="EPSG:4326"):
        """
        Converts the geometry to GeoJSON format.

        Parameters
        ----------
        crs : str, optional
            The CRS to convert to (default is EPSG:4326).

        Returns
        -------
        dict
            The geometry in GeoJSON format.
        """
        return json.loads(self._obj.to_crs(crs=crs).to_json(drop_id=True))

    def to_geopandas(self):
        """
        Converts the input geometry to a GeoDataFrame.

        Returns
        -------
        GeoDataFrame
            The input geometry as a GeoDataFrame.
        """
        return self._unknow_geometry_to_geodataframe(self.geometry)

    def to_bbox(self, crs="EPSG:4326"):
        """
        Returns the bounding box of the geometry.

        Parameters
        ----------
        crs : str, optional
            The CRS to convert to (default is EPSG:4326).

        Returns
        -------
        numpy.ndarray
            The bounding box as an array [minx, miny, maxx, maxy].
        """
        return self._obj.to_crs(crs=crs).total_bounds

    def _unknow_geometry_to_geodataframe(self, geometry):
        """
        Attempts to convert an unknown geometry format into a GeoDataFrame.

        Parameters
        ----------
        geometry : various types
            The input geometry, which can be a GeoDataFrame, GeoSeries, WKT string,
            or GeoJSON.

        Returns
        -------
        GeoDataFrame
            The converted geometry as a GeoDataFrame.

        Raises
        ------
        NotImplementedError
            If the geometry type cannot be inferred or converted.
        """
        if isinstance(geometry, gpd.GeoDataFrame):
            self.input_type = "GeoDataFrame"
            return geometry
        if isinstance(geometry, str):
            try:
                self.input_type = "wkt"
                return gpd.GeoDataFrame(geometry=[shapely.wkt.loads(geometry)], crs="EPSG:4326")
            except Exception:
                pass
        if isinstance(geometry, (dict, str)):
            self.input_type = "GeoJson"
            try:
                return gpd.read_file(geometry, driver="GeoJson", crs="EPSG:4326")
            except Exception:
                try:
                    return gpd.read_file(json.dumps(geometry), driver="GeoJson", crs="EPSG:4326")
                except Exception:
                    pass

            try:
                return gpd.GeoDataFrame.from_features(geometry, crs="EPSG:4326")
            except Exception:
                if "type" in geometry:
                    geom = shapely.__dict__[geometry["type"]]([geometry["coordinates"][0]])
                    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

        elif isinstance(geometry, gpd.GeoSeries):
            self.input_type = "GeoSeries"
            return gpd.GeoDataFrame(
                geometry=geometry,
                crs="EPSG:4326" if geometry.crs is None else geometry.crs,
            )
        else:
            raise NotImplementedError("Couldn't guess your geometry type")

    def buffer_in_meter(self, distance: int, crs_meters: str = "EPSG:3857", **kwargs):
        """
        Applies a buffer in meters to the geometry and returns it with the original CRS.

        Parameters
        ----------
        distance : int
            The buffer distance in meters.
        crs_meters : str, optional
            The CRS to use for calculating the buffer (default is EPSG:3857).
        **kwargs : dict, optional
            Additional keyword arguments to pass to the buffer method.

        Returns
        -------
        GeoSeries
            The buffered geometry in the original CRS.
        """
        return self._obj.to_crs(crs=crs_meters).buffer(distance=distance, **kwargs).to_crs(crs=self._obj.crs).geometry


def rasterize(gdf, datacube, all_touched=True, fill=0):
    gdf["geometry"] = gdf.to_crs(datacube.rio.crs).clip_by_rect(*datacube.rio.bounds())
    shapes = ((gdf.iloc[i].geometry, i + 1) for i in range(gdf.shape[0]))

    # rasterize features to use numpy/scipy to avoid polygon clipping
    feats = features.rasterize(
        shapes=shapes,
        fill=fill,
        out_shape=datacube.rio.shape,
        transform=datacube.rio.transform(),
        all_touched=all_touched,
    )
    return feats


def _compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def _indices_sparse(data):
    M = _compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


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
    if memory is None:
        memory = psutil.virtual_memory().available / 1e6
        logging.debug(f"Hoping to use a maximum memory {memory}Mo.")
    nbytes_per_date = int(dataset.nbytes / 1e6) / dataset.time.size * 3
    max_time_chunks = int(np.arange(0, memory, nbytes_per_date + 0.1).size)
    time_chunks = int(dataset.time.size / np.arange(0, dataset.time.size, max_time_chunks).size)
    logging.debug(f"Mo per date : {nbytes_per_date:0.2f}, total : {(nbytes_per_date*dataset.time.size):0.2f}.")
    logging.debug(f"Time chunks : {time_chunks} (on {dataset.time.size} time).")
    return time_chunks


def _zonal_stats_numpy(dataset: xr.Dataset, positions, reducers: list = ["mean"], all_touched=False):
    def _zonal_stats_ufunc(dataset, positions, reducers):
        zs = []
        for idx in range(len(positions)):
            field_stats = []
            for reducer in reducers:
                field_arr = dataset[(...,) + tuple(positions[idx])]
                if reducer == "mode":
                    field_arr = mode(field_arr, axis=-1, nan_policy="omit").mode
                else:
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
            "output_sizes": dict(feature=len(positions), zonal_statistics=len(reducers)),
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
            isel_time = np.arange(time_idx, np.min((time_idx + time_chunks, dataset.time.size)))
            ds = dataset.copy().isel(time=isel_time)
            if smart_load:
                t0 = time.time()
                ds = ds.load()
                logging.debug(f"Subdataset of {ds.time.size} dates loaded in memory in {(time.time()-t0):0.2f}s.")
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
        if "time" in dataset.dims and smart_load:
            time_chunks = _memory_time_chunks(dataset, memory)
            zs = xr.concat(
                [z for z in _loop_time_chunks(dataset, method, smart_load, time_chunks)],
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

        index = xr.DataArray(index, dims=["feature"], coords={"feature": zs.feature.values})

        # create the WKT geom
        if isinstance(buffer_meters, float | int):
            geoms.geometry = geoms["geometry_original"]

        if geoms.crs.to_epsg() != 4326:
            geoms = geoms.to_crs("EPSG:4326")

        geometry = xr.DataArray(
            geoms.geometry.iloc[list(f - 1)].to_wkt(rounding_precision=-1).values,
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


class NoItemsFoundError(Exception):
    """Exception raised when no items are found during search operation.

    This exception is raised when a parallel search operation yields no results,
    indicating that the search criteria did not match any items in the dataset.
    """

    pass


def datetime_to_str(dt_range: DatetimeRange) -> Tuple[str, str]:
    """Convert a datetime range to a tuple of formatted strings.

    Parameters
    ----------
    dt_range : tuple of (datetime or Timestamp)
        A tuple containing start and end datetimes to be converted.

    Returns
    -------
    tuple of str
        A tuple containing two strings representing the formatted start and end dates.

    Notes
    -----
    This function relies on ItemSearch._format_datetime internally to perform the
    actual formatting. The returned strings are split from a forward-slash separated
    string format.

    Examples
    --------
    >>> start = pd.Timestamp('2023-01-01')
    >>> end = pd.Timestamp('2023-12-31')
    >>> datetime_to_str((start, end))
    ('2023-01-01', '2023-12-31')
    """
    formatted = ItemSearch(url=None)._format_datetime(dt_range)
    start, end = formatted.split("/")
    return start, end


def datetime_split(
    dt_range: DatetimeRange, freq: Union[str, int, Timedelta] = "auto", n_jobs: int = 10
) -> Union[DatetimeRange, Tuple[DateRangeList, Timedelta]]:
    """Split a datetime range into smaller chunks based on specified frequency.

    Parameters
    ----------
    dt_range : tuple of (datetime or Timestamp)
        A tuple containing the start and end datetimes to split.
    freq : str or int or Timedelta, default="auto"
        The frequency to use for splitting the datetime range.
        If "auto", frequency is calculated based on the total date range:
        It increases by 5 days for every 6 months in the range.
        If int, interpreted as number of days.
        If Timedelta, used directly as the splitting frequency.
    n_jobs : int, default=10
        Number of jobs for parallel processing (currently unused in the function
        but maintained for API compatibility).

    Returns
    -------
    Union[DatetimeRange, tuple[list[DatetimeRange], Timedelta]]
        If the date range is smaller than the frequency:
            Returns the original datetime range tuple.
        Otherwise:
            Returns a tuple containing:
            - List of datetime range tuples split by the frequency
            - The Timedelta frequency used for splitting

    Notes
    -----
    The automatic frequency calculation uses the formula:
    freq = total_days // (5 + 5 * (total_days // 183))

    This ensures that the frequency increases by 5 days for every 6-month period
    in the total date range.

    Examples
    --------
    >>> start = pd.Timestamp('2023-01-01')
    >>> end = pd.Timestamp('2023-12-31')
    >>> splits, freq = datetime_split((start, end))
    >>> len(splits)  # Number of chunks
    12

    >>> # Using fixed frequency
    >>> splits, freq = datetime_split((start, end), freq=30)  # 30 days
    >>> freq
    Timedelta('30 days')
    """
    # Convert input dates to pandas Timestamps
    start, end = [pd.Timestamp(date) for date in datetime_to_str(dt_range)]
    date_diff = end - start

    # Calculate or convert frequency
    if freq == "auto":
        # Calculate automatic frequency based on total range
        total_days = date_diff.days
        months_factor = total_days // 183  # 183 days ≈ 6 months
        freq = Timedelta(days=(total_days // (5 + 5 * months_factor)))
    elif isinstance(freq, (int, str)):
        freq = Timedelta(days=int(freq))
    elif not isinstance(freq, Timedelta):
        raise TypeError("freq must be 'auto', int, or Timedelta")

    # Return original range if smaller than frequency
    if date_diff.days < freq.days or freq.days <= 1:
        return dt_range, freq

    # Generate date ranges
    date_ranges = [(chunk, min(chunk + freq, end)) for chunk in pd.date_range(start, end, freq=freq)[:-1]]

    return date_ranges, freq


def parallel_search(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for parallelizing search operations across datetime ranges.

    This decorator enables parallel processing of search operations by splitting the
    datetime range into batches. It automatically handles parallel execution when
    conditions are met (multiple batches or large date range) and falls back to
    sequential processing otherwise.

    Parameters
    ----------
    func : callable
        The search function to be parallelized. Should accept the following kwargs:
        - datetime : tuple of datetime
            Range of dates to search
        - batch_days : int or "auto", optional
            Number of days per batch for splitting
        - n_jobs : int, optional
            Number of parallel jobs. Use -1 or >10 for maximum of 10 jobs
        - raise_no_items : bool, optional
            Whether to raise exception when no items found

    Returns
    -------
    callable
        Wrapped function that handles parallel execution of the search operation.

    Notes
    -----
    The wrapped function preserves the same interface as the original function
    but adds parallel processing capabilities based on the following parameters
    in kwargs:
    - batch_days : Controls the size of datetime batches
    - n_jobs : Controls the number of parallel jobs (max 10)
    - datetime : Required for parallel execution

    The parallel execution uses threading backend from joblib.

    See Also
    --------
    joblib.Parallel : Used for parallel execution
    datetime_split : Helper function for splitting datetime ranges

    Examples
    --------
    >>> @parallel_search
    ... def search_items(query, datetime=None, batch_days="auto", n_jobs=1):
    ...     # Search implementation
    ...     return items
    >>>
    >>> # Will execute in parallel if conditions are met
    >>> items = search_items("query",
    ...                     datetime=(start_date, end_date),
    ...                     batch_days=30,
    ...                     n_jobs=4)
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()

        # Set default parameters
        batch_days = kwargs.setdefault("batch_days", "auto")
        n_jobs = kwargs.setdefault("n_jobs", -1)
        dt_range = kwargs.get("datetime")

        should_parallelize = _should_run_parallel(dt_range, batch_days, n_jobs)

        if should_parallelize:
            items = _run_parallel_search(func, args, kwargs, dt_range, batch_days)
        else:
            items = func(*args, **kwargs)

        execution_time = np.round(time.time() - start_time, 3)
        logging.info(f"Search/load items: {execution_time}s")

        return items

    return wrapper


def _should_run_parallel(dt_range: Optional[tuple[datetime, datetime]], batch_days: Any, n_jobs: int) -> bool:
    """Check if parallel execution should be used based on input parameters.
    Parameters
    ----------
    dt_range : tuple of datetime or None
        The start and end datetime for the search range
    batch_days : int or "auto" or None
        Number of days per batch for splitting the datetime range
    n_jobs : int
        Number of parallel jobs requested
    Returns
    -------
    bool
        True if parallel execution should be used, False otherwise
    Notes
    -----
    Parallel execution is used when all of the following conditions are met:
    - dt_range is provided and not None
    - batch_days is not None
    - n_jobs > 1
    - Either multiple date ranges exist or the total days exceed batch_days
    """
    # Check for basic conditions that prevent parallel execution
    if not dt_range or batch_days is None or n_jobs <= 1:
        return False

    # Split the datetime range
    date_ranges, freq = datetime_split(dt_range, batch_days)

    # Check if splitting provides meaningful parallelization
    delta_days = (date_ranges[-1][-1] - date_ranges[0][0]).days
    return len(date_ranges) > 1 or delta_days > batch_days


def _run_parallel_search(
    func: Callable,
    args: tuple,
    kwargs: dict,
    dt_range: tuple[datetime, datetime],
    batch_days: Any,
) -> T:
    """Execute the search function in parallel across datetime batches.

    Parameters
    ----------
    func : callable
        The search function to be executed in parallel
    args : tuple
        Positional arguments to pass to the search function
    kwargs : dict
        Keyword arguments to pass to the search function
    dt_range : tuple of datetime
        The start and end datetime for the search range
    batch_days : int or "auto"
        Number of days per batch for splitting the datetime range

    Returns
    -------
    T
        Combined results from all parallel executions

    Raises
    ------
    NoItemsFoundError
        If no items are found across all parallel executions

    Notes
    -----
    This function:
    1. Splits the datetime range into batches
    2. Configures parallel execution parameters
    3. Runs the search function in parallel using joblib
    4. Combines results from all parallel executions

    The maximum number of parallel jobs is capped at 10, and -1 is converted to 10.
    """
    date_ranges, freq = datetime_split(dt_range, batch_days)

    logging.info(f"Search parallel with {kwargs['n_jobs']} jobs, split every {freq.days} days.")

    # Prepare kwargs for parallel execution
    parallel_kwargs = kwargs.copy()
    parallel_kwargs.pop("datetime")
    parallel_kwargs["raise_no_items"] = False

    # Handle n_jobs special case: -1 should become 10
    n_jobs = parallel_kwargs.get("n_jobs", 10)
    parallel_kwargs["n_jobs"] = 10 if (n_jobs == -1 or n_jobs > 10) else n_jobs

    # Execute parallel search
    results = Parallel(n_jobs=parallel_kwargs["n_jobs"], backend="threading")(
        delayed(func)(*args, datetime=dt, **parallel_kwargs) for dt in date_ranges
    )

    # Combine results
    items = ItemCollection(chain(*results))
    if not items:
        raise NoItemsFoundError("No items found in parallel search")

    return items


def apply_single_condition(item_value, condition_op: str, condition_value: list[any, list[any]]) -> bool:
    """
    Apply a single comparison condition to an item's property value.

    Parameters
    ----------
    item_value : any
        The value of the property in the item.
    condition_op : str
        The comparison operator (e.g., 'lt', 'gt', 'eq').
    condition_value : [any, list[any]]
        The value or list of values to compare against.

    Returns
    -------
    bool
        True if the condition is met, False otherwise.
    """
    # Ensure condition_value is always a list
    values = condition_value if isinstance(condition_value, list) else [condition_value]

    # Get the comparison function from the operator module
    op_func = operator.__dict__.get(condition_op)
    if not op_func:
        raise ValueError(f"Unsupported operator: {condition_op}")

    # Check if any value meets the condition
    return any(op_func(item_value, val) for val in values)


def validate_property_condition(item: any, property_name: str, conditions: dict[str, any]) -> bool:
    """
    Validate if an item meets all conditions for a specific property.

    Parameters
    ----------
    item : any
        The STAC item to check.
    property_name : str
        The name of the property to validate.
    conditions : dict[str, any]
        Dictionary of conditions to apply to the property.

    Returns
    -------
    bool
        True if all conditions are met, False otherwise.
    """
    # Check if the property exists in the item
    if property_name not in item.properties:
        return False

    # Check each condition for the property
    return all(
        apply_single_condition(item.properties.get(property_name), condition_op, condition_value)
        for condition_op, condition_value in conditions.items()
    )


def filter_items(items: list[any], query: dict[str, dict[str, any]]) -> list[any]:
    """
    Filter items based on a complex query dictionary.

    Parameters
    ----------
    items : list[any]
        List of STAC items to filter.
    query : dict[str, dict[str, any]]
        Query filter with operations to apply to item properties.

    Returns
    -------
    list[any]
        Filtered list of items matching the query.

    Examples
    --------
    >>> query = {
    ...     'eo:cloud_cover': {'lt': [10], 'gt': [0]},
    ...     'datetime': {'eq': '2023-01-01'}
    ... }
    >>> filtered_items = filter_items(catalog_items, query)
    """
    return [
        item
        for item in items
        if all(
            validate_property_condition(item, property_name, conditions) for property_name, conditions in query.items()
        )
    ]


def post_query_items(items: list[any], query: dict[str, dict[str, any]]) -> ItemCollection:
    """
    Apply a query filter to items fetched from a STAC catalog and return an ItemCollection.

    Parameters
    ----------
    items : list[any]
        List of STAC items to filter.
    query : dict[str, dict[str, any]]
        Query filter with operations to apply to item properties.

    Returns
    -------
    ItemCollection
        Filtered collection of items matching the query.

    Examples
    --------
    >>> query = {
    ...     'eo:cloud_cover': {'lt': [10], 'gt': [0]},
    ...     'datetime': {'eq': '2023-01-01'}
    ... }
    >>> filtered_items = post_query_items(catalog_items, query)
    """
    filtered_items = filter_items(items, query)
    return ItemCollection(filtered_items)  # Assuming ItemCollection is imported/defined elsewhere


def _select_last_common_occurrences(first, second):
    """
    For each date in second dataset, select the last N occurrences of that date from first dataset,
    where N is the count of that date in second dataset.

    Parameters:
    first (xarray.Dataset): Source dataset
    second (xarray.Dataset): Dataset containing the dates to match and their counts

    Returns:
    xarray.Dataset: Subset of first dataset with selected time indices
    """
    # Convert times to datetime64[ns] if they aren't already
    first_times = first.time.astype("datetime64[ns]")
    second_times = second.time.astype("datetime64[ns]")

    # Get unique dates and their counts from second dataset
    unique_dates, counts = np.unique(second_times.values, return_counts=True)

    # Initialize list to store selected indices
    selected_indices = []

    # For each unique date in second
    for date, count in zip(unique_dates, counts):
        # Find all indices where this date appears in first
        date_indices = np.where(first_times == date)[0]
        # Take the last 'count' number of indices
        selected_indices.extend(date_indices[-count:])

    # Sort indices to maintain temporal order (or reverse them if needed)
    selected_indices = sorted(selected_indices, reverse=True)

    # Select these indices from the first dataset
    return first.isel(time=selected_indices)


def _cloud_path_to_http(cloud_path):
    """Convert a cloud path to HTTP URL.

    Parameters
    ----------
    cloud_path : str
        Cloud path

    Returns
    -------
    url : str
        HTTP URL
    """
    endpoints = dict(s3="s3.amazonaws.com")
    cloud_provider = cloud_path.split("://")[0]
    container = cloud_path.split("/")[2]
    key = "/".join(cloud_path.split("/")[3:])
    endpoint = endpoints.get(cloud_provider, None)
    if endpoint:
        url = f"https://{container}.{endpoint}/{key}"
    else:
        url = cloud_path
    return url


def enhance_assets(
    items: ItemCollection,
    alternate: str = "download",
    use_http_url: bool = False,
    add_default_scale_factor: bool = False,
) -> ItemCollection:
    """
    Enhance STAC item assets with additional metadata and URL transformations.

    Parameters
    ----------
    items : ItemCollection
        Collection of STAC items to enhance
    alternate : Optional[str], optional
        Alternate asset href to use, by default "download"
    use_http_url : bool, optional
        Convert cloud URLs to HTTP URLs, by default False
    add_default_scale_factor : bool, optional
        Add default scale, offset, nodata to raster bands, by default False

    Returns
    -------
    ItemCollection
        Enhanced collection of STAC items
    """
    if any((alternate, use_http_url, add_default_scale_factor)):
        for idx, item in enumerate(items):
            keys = list(item.assets.keys())
            for asset in keys:
                # use the alternate href if it exists
                if alternate:
                    href = item.assets[asset].extra_fields.get("alternate", {}).get(alternate, {}).get("href")
                    if href:
                        items[idx].assets[asset].href = href
                # use HTTP URL instead of cloud path
                if use_http_url:
                    href = item.assets[asset].to_dict().get("href", {})
                    if href:
                        items[idx].assets[asset].href = _cloud_path_to_http(href)
                if add_default_scale_factor:
                    scale_factor_collection = scale_factor_collections.get(item.collection_id, [{}])
                    for scales_collection in scale_factor_collection:
                        if asset in scales_collection.get("assets", []):
                            if "raster:bands" not in items[idx].assets[asset].extra_fields:
                                items[idx].assets[asset].extra_fields["raster:bands"] = [{}]
                            if not items[idx].assets[asset].extra_fields["raster:bands"][0].get("scale"):
                                items[idx].assets[asset].extra_fields["raster:bands"][0]["scale"] = scales_collection[
                                    "scale"
                                ]
                                items[idx].assets[asset].extra_fields["raster:bands"][0]["offset"] = scales_collection[
                                    "offset"
                                ]
                                items[idx].assets[asset].extra_fields["raster:bands"][0]["nodata"] = scales_collection[
                                    "nodata"
                                ]

    return items


class StacCollectionExplorer:
    """
    A class to explore a STAC collection.

    Parameters
    ----------
    client : Client
        A PySTAC client for interacting with the Earth Data Store STAC API.
    collection : str
        The name of the collection to explore.

    Returns
    -------
    None
    """

    def __init__(self, client, collection):
        self.client = client
        self.collection = collection
        self.client_collection = self.client.get_collection(self.collection)
        self.item = self.__first_item()
        self.properties = self.client_collection.to_dict()

    def __first_item(self):
        """Get the first item of the STAC collection as an overview of the items content.

        Returns
        -------
        item : Item
            The first item of the collection.
        """
        for item in self.client.get_collection(self.collection).get_items():
            self.item = item
            break
        return self.item

    @property
    def item_properties(self):
        return {k: self.item.properties[k] for k in sorted(self.item.properties.keys())}

    def assets(self, asset_name=None):
        if asset_name:
            return self.asset_metadata(asset_name)
        return list(sorted(self.item.assets.keys()))

    def assets_metadata(self, asset_name=None):
        if asset_name:
            return self.asset_metadata(asset_name)
        return {k: self.asset_metadata(k) for k in self.assets()}

    def asset_metadata(self, asset_name):
        return self.item.assets[asset_name].to_dict()

    def __repr__(self):
        return f'Exploring collection "{self.collection}"'


class AgricultureServiceV0:
    def __init__(self, config: str | dict = None, presign_urls=True, asset_proxy_enabled=False):
        """
        A client for interacting with the Earth Data Store API.
        By default, Earth Data Store will look for environment variables called
        EDS_AUTH_URL, EDS_SECRET and EDS_CLIENT_ID.

        Parameters
        ----------
        config : str | dict, optional
            The path to the json file containing the Earth Data Store credentials,
            or a dict with those credentials.
        asset_proxy_enabled : bool, optional
            Use asset proxy URLs, by default False

        Returns
        -------
        None.

        Example
        --------
        >>> eds = earthdaily.earthdatastore()
        >>> collection = "venus-l2a"
        >>> theia_location = "MEAD"
        >>> max_cloud_cover = 20
        >>> query = { "theia:location": {"eq": theia_location}, "eo:cloud_cover": {"lt": max_cloud_cover} }
        >>> items = eds.search(collections=collection, query=query)
        >>> print(len(items))
        132
        """
        if not isinstance(config, dict):
            warnings.warn(
                "Using directly the Auth class to load credentials is deprecated. "
                "Please use earthdaily.EarthDataStore() instead",
                FutureWarning,
            )

        self._client = None
        self.__auth_config = config
        self.__presign_urls = presign_urls
        self.__asset_proxy_enabled = asset_proxy_enabled
        self._first_items_ = {}
        self._staccollectionexplorer = {}
        self.__time_eds_log = time.time()
        self._client = self.client

    @classmethod
    def from_credentials(
        cls,
        json_path: Optional[Path] = None,
        toml_path: Optional[Path] = None,
        profile: Optional[str] = None,
        presign_urls: bool = True,
        asset_proxy_enabled: bool = False,
    ) -> "AgricultureServiceV0":
        """
        Secondary Constructor.
        Try to read Earth Data Store credentials from multiple sources, in the following order:
            - from input credentials stored in a given JSON file
            - from input credentials stored in a given TOML file
            - from environement variables
            - from the $HOME/.earthdaily/credentials TOML file and a given profile
            - from the $HOME/.earthdaily/credentials TOML file and the "default" profile

        Parameters
        ----------
        path : Path, optional
            The path to the TOML file containing the Earth Data Store credentials.
            Uses "$HOME/.earthdaily/credentials" by default.
        profile : profile, optional
            Name of the profile to use in the TOML file.
            Uses "default" by default.
        asset_proxy_enabled : bool, optional
            Use asset proxy URLs, by default False

        Returns
        -------
        Auth
            A :class:`Auth` instance
        """
        config = cls.read_credentials(
            json_path=json_path,
            toml_path=toml_path,
            profile=profile,
        )

        for item, value in config.items():
            if not value:
                raise ValueError(f"Missing value for {item}")

        return cls(
            config=config,
            presign_urls=presign_urls,
            asset_proxy_enabled=asset_proxy_enabled,
        )

    @classmethod
    def read_credentials(
        cls,
        json_path: Optional[Path] = None,
        toml_path: Optional[Path] = None,
        profile: Optional[str] = None,
    ) -> "AgricultureServiceV0":
        """
        Try to read Earth Data Store credentials from multiple sources, in the following order:
            - from input credentials stored in a given JSON file
            - from input credentials stored in a given TOML file
            - from environement variables
            - from the $HOME/.earthdaily/credentials TOML file and a given profile
            - from the $HOME/.earthdaily/credentials TOML file and the "default" profile

        Parameters
        ----------
        path : Path, optional
            The path to the TOML file containing the Earth Data Store credentials.
            Uses "$HOME/.earthdaily/credentials" by default.
        profile : profile, optional
            Name of the profile to use in the TOML file.
            Uses "default" by default.

        Returns
        -------
        dict
            Dictionary containing credentials
        """
        if json_path is not None:
            config = cls.read_credentials_from_json(json_path=json_path)

        elif toml_path is not None:
            config = cls.read_credentials_from_toml(toml_path=toml_path, profile=profile)

        elif os.getenv("EDS_AUTH_URL") and os.getenv("EDS_SECRET") and os.getenv("EDS_CLIENT_ID"):
            config = cls.read_credentials_from_environment()

        else:
            config = cls.read_credentials_from_ini()

        return config

    @classmethod
    def read_credentials_from_ini(cls, profile: str = "default") -> dict:
        """
        Read Earth Data Store credentials from a ini file.

        Parameters
        ----------
        ini_path : Path
            The path to the INI file containing the Earth Data Store credentials.
        Returns
        -------
        dict
           Dictionary containing credentials
        """

        from configparser import ConfigParser

        ini_path = Path.home() / ".earthdaily/credentials"
        ini_config = ConfigParser()
        ini_config.read(ini_path)
        ini_config = ini_config[profile]
        config = {key.upper(): value for key, value in ini_config.items()}
        return config

    @classmethod
    def read_credentials_from_json(cls, json_path: Path) -> dict:
        """
        Read Earth Data Store credentials from a JSON file.

        Parameters
        ----------
        json_path : Path
            The path to the JSON file containing the Earth Data Store credentials.
        Returns
        -------
        dict
           Dictionary containing credentials
        """
        if isinstance(json_path, dict):
            return json_path
        with json_path.open() as file_object:
            config = json.load(file_object)
        return config

    @classmethod
    def read_credentials_from_environment(cls) -> dict:
        """
        Read Earth Data Store credentials from environment variables.

        Returns
        -------
        dict
            Dictionary containing credentials
        """
        config = {
            "EDS_AUTH_URL": os.getenv("EDS_AUTH_URL"),
            "EDS_SECRET": os.getenv("EDS_SECRET"),
            "EDS_CLIENT_ID": os.getenv("EDS_CLIENT_ID"),
        }

        # Optional
        if "EDS_API_URL" in os.environ:
            config["EDS_API_URL"] = os.getenv("EDS_API_URL")

        return config

    @classmethod
    def read_credentials_from_toml(cls, toml_path: Path = None, profile: str = None) -> dict:
        """
        Read Earth Data Store credentials from a TOML file

        Parameters
        ----------
        toml_path : Path, optional
            The path to the TOML file containing the Earth Data Store credentials.
        profile : profile, optional
            Name of the profile to use in the TOML file

        Returns
        -------
        dict
            Dictionary containing credentials
        """
        if not toml_path.exists():
            raise FileNotFoundError(f"Credentials file {toml_path} not found. Make sure the path is valid")

        with toml_path.open() as f:
            config = toml.load(f)

        if profile not in config:
            raise ValueError(f"Credentials profile {profile} not found in {toml_path}")

        config = config[profile]

        return config

    def get_access_token(self, config: Optional[EarthDataStoreConfig] = None):
        """
        Retrieve an access token for interacting with the EarthDataStore API.

        By default, the method will look for environment variables:
        EDS_AUTH_URL, EDS_SECRET, and EDS_CLIENT_ID. Alternatively, a configuration
        object or dictionary can be passed to override these values.

        Parameters
        ----------
        config : EarthDataStoreConfig, optional
            A configuration object containing the Earth Data Store credentials.

        Returns
        -------
        str
            The access token for authenticating with the Earth Data Store API.
        """
        if not config:
            config = self._config_parser(self.__auth_config)
        response = requests.post(
            config.auth_url,
            data={"grant_type": "client_credentials"},
            allow_redirects=False,
            auth=(config.client_id, config.client_secret),
        )
        response.raise_for_status()
        return response.json()["access_token"]

    def _config_parser(self, config=None):
        """
        Parse and construct the EarthDataStoreConfig object from various input formats.

        The method supports configuration as a dictionary, JSON file path, tuple,
        or environment variables.

        Parameters
        ----------
        config : dict or str or tuple, optional
            Configuration source. It can be:
            - A dictionary containing the required API credentials.
            - A string path to a JSON file containing these credentials.
            - A tuple of (access_token, eds_url).
            - None, in which case environment variables will be used.

        Returns
        -------
        EarthDataStoreConfig
            A configuration object containing the required API credentials.

        Raises
        ------
        AttributeError
            If required credentials are missing in the provided input or environment variables.
        """
        if isinstance(config, tuple):  # token
            access_token, eds_url = config
            logging.log(level=logging.INFO, msg="Using token to reauth")
            return EarthDataStoreConfig(eds_url=eds_url, access_token=access_token)
        else:
            if isinstance(config, dict):
                config = config.get
            elif isinstance(config, str) and config.endswith(".json"):
                config = json.load(open(config, "rb")).get

            if config is None:
                config = os.getenv
            auth_url = config("EDS_AUTH_URL")
            client_secret = config("EDS_SECRET")
            client_id = config("EDS_CLIENT_ID")
            eds_url = config("EDS_API_URL", "https://api.earthdaily.com/platform/v1/stac")
            if auth_url is None or client_secret is None or client_id is None:
                raise AttributeError("You need to have env : EDS_AUTH_URL, EDS_SECRET and EDS_CLIENT_ID")
            return EarthDataStoreConfig(
                auth_url=auth_url,
                client_secret=client_secret,
                client_id=client_id,
                eds_url=eds_url,
            )

    def _get_client(self, config=None, presign_urls=True, asset_proxy_enabled=False):
        """Get client for interacting with the EarthDataStore API.

        By default, Earth Data Store will look for environment variables called
        EDS_AUTH_URL, EDS_SECRET and EDS_CLIENT_ID.

        Parameters
        ----------
        config : str | dict, optional
            A JSON string or a dictionary with the credentials for the Earth Data Store.
        presign_urls : bool, optional
            Use presigned URLs, by default True
        asset_proxy_enabled : bool, optional
            Use asset proxy URLs, by default False

        Returns
        -------
        client : Client
            A PySTAC client for interacting with the Earth Data Store STAC API.

        """

        config = self._config_parser(config)

        if config.access_token:
            access_token = config.access_token
        else:
            access_token = self.get_access_token(config)

        headers = {"Authorization": f"bearer {access_token}"}
        if asset_proxy_enabled:
            headers["X-Proxy-Asset-Urls"] = "True"
        elif presign_urls:
            headers["X-Signed-Asset-Urls"] = "True"

        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=None,
        )
        stac_api_io = StacApiIO(max_retries=retry)

        return Client.open(config.eds_url, headers=headers, stac_io=stac_api_io)

    @property
    def client(self) -> Client:
        """
        Create an instance of pystac client from EarthDataSTore

        Returns
        -------
            A :class:`Client` instance for this Catalog.
        """
        if t := (time.time() - self.__time_eds_log) > 3600 or self._client is None:
            if t:
                logging.log(level=logging.INFO, msg="Reauth to EarthDataStore")
            self._client = self._get_client(self.__auth_config, self.__presign_urls, self.__asset_proxy_enabled)
            self.__time_eds_log = time.time()

        return self._client

    def explore(self, collection: str = None):
        """
        Explore a collection, its properties and assets. If not collection specified,
        returns the list of collections.

        Parameters
        ----------
        collection : str, optional.
            Collection name. The default is None.

        Returns
        -------
        str|StacCollectionExplorer
            The list of collections, or a collection to explore using module
            StacCollectionExplorer.

        Example
        --------
        >>> eds = earthdaily.earthdatastore()
        >>> collection = "venus-l2a"
        >>> eds.explore(collection).item_properties
        {'constellation': 'VENUS',
         'created': '2023-06-14T00:14:10.167450Z',
         'datetime': '2023-06-07T11:23:18.000000Z',
         'description': '',
         'eda:geometry_tags': ['RESOLVED_CLOCKWISE_POLYGON'],
         'eda:loose_validation_status': 'VALID',
         'eda:num_cols': 9106,
         'eda:num_rows': 11001,
         'eda:original_geometry': {'type': 'Polygon',
          'coordinates': [[[-16.684545516968, 16.109294891357],
            [-16.344039916992, 16.111709594727],
            [-16.341398239136, 15.714001655579],
            [-16.68123626709, 15.711649894714],
            [-16.684545516968, 16.109294891357]]]},
         'eda:product_type': 'REFLECTANCE',
         'eda:sensor_type': 'OPTICAL',
         'eda:source_created': '2023-06-13T18:47:27.000000Z',
         'eda:source_updated': '2023-06-13T20:22:35.000000Z',
         'eda:status': 'PUBLISHED',
         'eda:tracking_id': 'MutZbYe54RY7eP3iuAbtKb',
         'eda:unusable_cover': 0.0,
         'eda:water_cover': 0.0,
         'end_datetime': '2023-06-07T11:23:18.000000Z',
         'eo:bands': [{'name': 'B1',
           'common_name': 'coastal',
           'description': 'B1',
           'center_wavelength': 0.424},
          {'name': 'B2',
           'common_name': 'coastal',
           'description': 'B2',
           'center_wavelength': 0.447},
          {'name': 'B3',
           'common_name': 'blue',
           'description': 'B3',
           'center_wavelength': 0.492},
          {'name': 'B4',
           'common_name': 'green',
           'description': 'B4',
           'center_wavelength': 0.555},
          {'name': 'B5',
           'common_name': 'yellow',
           'description': 'B5',
           'center_wavelength': 0.62},
          {'name': 'B6',
           'common_name': 'yellow',
           'description': 'B6',
           'center_wavelength': 0.62},
          {'name': 'B7',
           'common_name': 'red',
           'description': 'B7',
           'center_wavelength': 0.666},
          {'name': 'B8',
           'common_name': 'rededge',
           'description': 'B8',
           'center_wavelength': 0.702},
          {'name': 'B9',
           'common_name': 'rededge',
           'description': 'B9',
           'center_wavelength': 0.741},
          {'name': 'B10',
           'common_name': 'rededge',
           'description': 'B10',
           'center_wavelength': 0.782},
          {'name': 'B11',
           'common_name': 'nir08',
           'description': 'B11',
           'center_wavelength': 0.861},
          {'name': 'B12',
           'common_name': 'nir09',
           'description': 'B12',
           'center_wavelength': 0.909}],
         'eo:cloud_cover': 0.0,
         'gsd': 4.0,
         'instruments': ['VENUS'],
         'license': 'CC-BY-NC-4.0',
         'mission': 'venus',
         'platform': 'VENUS',
         'processing:level': 'L2A',
         'proj:epsg': 32628,
         'providers': [{'name': 'Theia',
           'roles': ['licensor', 'producer', 'processor']},
          {'url': 'https://earthdaily.com',
           'name': 'EarthDaily Analytics',
           'roles': ['processor', 'host']}],
         'sat:absolute_orbit': 31453,
         'start_datetime': '2023-06-07T11:23:18.000000Z',
         'theia:location': 'STLOUIS',
         'theia:product_id': 'VENUS-XS_20230607-112318-000_L2A_STLOUIS_C_V3-1',
         'theia:product_version': '3.1',
         'theia:publication_date': '2023-06-13T18:08:10.205000Z',
         'theia:sensor_mode': 'XS',
         'theia:source_uuid': 'a29bfc89-8372-5e91-841c-b11cdb40bb14',
         'title': 'VENUS-XS_20230607-112318-000_L2A_STLOUIS_D',
         'updated': '2023-06-14T00:42:17.898993Z',
         'view:azimuth': 33.293623499999995,
         'view:incidence_angle': 14.6423245,
         'view:sun_azimuth': 69.8849963957,
         'view:sun_elevation': 65.0159541684}

        """
        if collection:
            if collection not in self._staccollectionexplorer.keys():
                self._staccollectionexplorer[collection] = StacCollectionExplorer(self.client, collection)
            return self._staccollectionexplorer.get(collection)
        return sorted(c.id for c in self.client.get_all_collections())

    def _update_search_kwargs_for_ag_cloud_mask(
        self,
        search_kwargs,
        collections,
        key="eda:ag_cloud_mask_available",
        target_param="query",
    ):
        """Update the STAC search kwargs to only get items that have an available agricultural cloud mask.

        Args:
            search_kwargs (dict): The search kwargs to be updated.
            collections (str | list): The collection(s) to search.

        Returns:
            dict: The updated search kwargs.
        """
        search_kwargs = search_kwargs.copy()
        # to get only items that have a ag_cloud_mask
        ag_query = {key: {"eq": True}}

        # to check if field is queryable
        # =============================================================================
        #         queryables = self.client._stac_io.request(
        #             self.client.get_root_link().href
        #             + f"/queryables?collections={collections[0] if isinstance(collections,list) else collections}"
        #         )
        #         queryables = json.loads(queryables)
        #         queryables = queryables["properties"]
        #         if "eda:ag_cloud_mask_available" not in queryables.keys():
        #             target_param = "post_query"
        #         else:
        #             target_param = "query"
        # =============================================================================
        query = search_kwargs.get("target_param", {})
        query.update(ag_query)
        search_kwargs[target_param] = query
        return search_kwargs

    @_datacubes
    def datacube(
        self,
        collections: str | list,
        datetime=None,
        assets: None | list | dict = None,
        intersects: (gpd.GeoDataFrame | str | dict) = None,
        bbox=None,
        mask_with: (None | str | list) = None,
        mask_statistics: bool | int = False,
        clear_cover: (int | float) = None,
        prefer_alternate: (str | bool) = "download",
        search_kwargs: dict = {},
        add_default_scale_factor: bool = True,
        common_band_names=True,
        cross_calibration_collection: (None | str) = None,
        properties: (bool | str | list) = False,
        groupby_date: str = "mean",
        cloud_search_kwargs={},
        **kwargs,
    ) -> xr.Dataset:
        """
        Create a datacube.

        Parameters
        ----------
        collections : str | list
            If several collections, the first collection will be the reference collection (for spatial resolution).
        datetime: Either a single datetime or datetime range used to filter results.
            You may express a single datetime using a :class:`datetime.datetime`
            instance, a `RFC 3339-compliant <https://tools.ietf.org/html/rfc3339>`__
            timestamp, or a simple date string (see below). Instances of
            :class:`datetime.datetime` may be either
            timezone aware or unaware. Timezone aware instances will be converted to
            a UTC timestamp before being passed
            to the endpoint. Timezone unaware instances are assumed to represent UTC
            timestamps. You may represent a
            datetime range using a ``"/"`` separated string as described in the
            spec, or a list, tuple, or iterator
            of 2 timestamps or datetime instances. For open-ended ranges, use either
            ``".."`` (``'2020-01-01:00:00:00Z/..'``,
            ``['2020-01-01:00:00:00Z', '..']``) or a value of ``None``
            (``['2020-01-01:00:00:00Z', None]``).

            If using a simple date string, the datetime can be specified in
            ``YYYY-mm-dd`` format, optionally truncating
            to ``YYYY-mm`` or just ``YYYY``. Simple date strings will be expanded to
            include the entire time period, for example:

            - ``2017`` expands to ``2017-01-01T00:00:00Z/2017-12-31T23:59:59Z``
            - ``2017-06`` expands to ``2017-06-01T00:00:00Z/2017-06-30T23:59:59Z``
            - ``2017-06-10`` expands to
              ``2017-06-10T00:00:00Z/2017-06-10T23:59:59Z``

            If used in a range, the end of the range expands to the end of that
            day/month/year, for example:

            - ``2017/2018`` expands to
              ``2017-01-01T00:00:00Z/2018-12-31T23:59:59Z``
            - ``2017-06/2017-07`` expands to
              ``2017-06-01T00:00:00Z/2017-07-31T23:59:59Z``
            - ``2017-06-10/2017-06-11`` expands to
              ``2017-06-10T00:00:00Z/2017-06-11T23:59:59Z``
        assets : None | list | dict, optional
            DESCRIPTION. The default is None.
        intersects : (gpd.GeoDataFrame, str(wkt), dict(json)), optional
            DESCRIPTION. The default is None.
        bbox : TYPE, optional
            DESCRIPTION. The default is None.
        mask_with : (None, str, list), optional
            "native" mask, or "ag_cloud_mask", or ["ag_cloud_mask","native"],
            and so if ag_cloud_mask is not available, will switch to native.
            The default is None.
        mask_statistics : bool | int, optional
            DESCRIPTION. The default is False.
        clear_cover : (int, float), optional
            Percent of clear data above a field (from 0 to 100).
            The default is None.
        prefer_alternate : (str, False), optional
            Uses the alternate/download href instead of the default href.
            The default is "download".
        search_kwargs : dict, optional
            DESCRIPTION. The default is {}.
        add_default_scale_factor : bool, optional
            DESCRIPTION. The default is True.
        common_band_names : TYPE, optional
            DESCRIPTION. The default is True.
        cross_calibration_collection : (None | str), optional
            DESCRIPTION. The default is None.
        properties : (bool | str | list), optional
            Retrieve properties per item. The default is False.
        **kwargs : TYPE
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.
        Warning
            DESCRIPTION.

        Returns
        -------
        xr_datacube : TYPE
            DESCRIPTION.

        """

        # Properties (per items) are not compatible with groupby_date.
        if properties not in (None, False) and groupby_date is not None:
            raise NotImplementedError("You must set `groupby_date=None` to have properties per item.")

        # convert collections to list
        collections = [collections] if isinstance(collections, str) else collections

        # if intersects a geometry, create a GeoDataFRame
        if intersects is not None:
            intersects = GeometryManager(intersects).to_geopandas()
            self.intersects = intersects

        # if mask_with, need to add assets or to get mask item id
        if mask_with:
            if mask_with not in _available_masks:
                raise NotImplementedError(
                    f"Specified mask '{mask_with}' is not available. Available masks providers are : {_available_masks}"
                )

            elif mask_with in ["ag_cloud_mask", "agriculture-cloud-mask"]:
                search_kwargs = self._update_search_kwargs_for_ag_cloud_mask(
                    search_kwargs, collections[0], key="eda:ag_cloud_mask_available"
                )
                mask_with = "ag_cloud_mask"
            elif mask_with in [
                "cloud_mask",
                "cloudmask",
                "cloud_mask_ag_version",
                "cloudmask_ag_version",
            ]:
                search_kwargs = self._update_search_kwargs_for_ag_cloud_mask(
                    search_kwargs,
                    collections[0],
                    key="eda:cloud_mask_available",
                )

                mask_with = "cloud_mask"

            else:
                mask_with = _native_mask_def_mapping.get(collections[0], None)
                sensor_mask = _native_mask_asset_mapping.get(collections[0], None)

                if isinstance(assets, list) and sensor_mask not in assets:
                    assets.append(sensor_mask)
                elif isinstance(assets, dict):
                    assets[sensor_mask] = sensor_mask
        bbox_query = None

        if bbox is None and intersects is not None:
            bbox_query = list(GeometryManager(intersects).to_bbox())
        elif bbox is not None and intersects is None:
            bbox_query = bbox

        # query the items
        items = self.search(
            collections=collections,
            bbox=bbox_query,
            datetime=datetime,
            assets=assets,
            prefer_alternate=prefer_alternate,
            add_default_scale_factor=add_default_scale_factor,
            **search_kwargs,
        )

        xcal_items = None
        if isinstance(cross_calibration_collection, str) and cross_calibration_collection != collections[0]:
            try:
                xcal_items = self.search(
                    collections="eda-cross-calibration",
                    intersects=intersects,
                    query={
                        "eda_cross_cal:source_collection": {"eq": collections[0]},
                        "eda_cross_cal:destination_collection": {"eq": cross_calibration_collection},
                    },
                )
            except Warning:
                raise Warning("No cross calibration coefficient available for the specified collections.")

        # Create datacube from items
        xr_datacube = datacube(
            items,
            intersects=intersects,
            bbox=bbox,
            assets=assets,
            common_band_names=common_band_names,
            cross_calibration_items=xcal_items,
            properties=properties,
            groupby_date=None,
            **kwargs,
        )
        if intersects is not None:
            xr_datacube = xr_datacube.ed.clip(intersects)
        # Create mask datacube and apply it to xr_datacube
        if mask_with:
            if "geobox" not in kwargs:
                kwargs["geobox"] = xr_datacube.odc.geobox
            kwargs.pop("crs", "")
            kwargs.pop("resolution", "")
            kwargs["dtype"] = "int8"
            if clear_cover and mask_statistics is False:
                mask_statistics = True
            mask_kwargs = dict(mask_statistics=False)
            if mask_with == "ag_cloud_mask" or mask_with == "cloud_mask":
                mask_asset_mapping = {
                    "ag_cloud_mask": {"agriculture-cloud-mask": "ag_cloud_mask"},
                    "cloud_mask": {"cloud-mask": "cloud_mask"},
                }
                acm_items = self.find_cloud_mask_items(items, cloudmask=mask_with, **cloud_search_kwargs)
                acm_datacube = datacube(
                    acm_items,
                    intersects=intersects,
                    bbox=bbox,
                    groupby_date=None,
                    assets=mask_asset_mapping[mask_with],
                    **kwargs,
                )
                xr_datacube["time"] = xr_datacube.time.astype("M8[ns]")
                if xr_datacube.time.size != acm_datacube.time.size:
                    xr_datacube = _select_last_common_occurrences(xr_datacube, acm_datacube)
                acm_datacube["time"] = xr_datacube["time"].time
                acm_datacube = _match_xy_dims(acm_datacube, xr_datacube)
                xr_datacube = xr.merge((xr_datacube, acm_datacube), compat="override")

                # mask_kwargs.update(acm_datacube=acm_datacube)
            else:
                mask_assets = {_native_mask_asset_mapping[collections[0]]: _native_mask_def_mapping[collections[0]]}

                clouds_datacube = datacube(
                    items,
                    groupby_date=None,
                    bbox=list(GeometryManager(intersects).to_bbox()),
                    assets=mask_assets,
                    resampling=0,
                    **kwargs,
                )
                clouds_datacube = _match_xy_dims(clouds_datacube, xr_datacube)
                if intersects is not None:
                    clouds_datacube = clouds_datacube.ed.clip(intersects)
                xr_datacube = xr.merge((xr_datacube, clouds_datacube), compat="override")

            Mask_x = Mask(xr_datacube, intersects=intersects, bbox=bbox)
            xr_datacube = getattr(Mask_x, mask_with)(**mask_kwargs)

        # keep only one value per pixel per day
        if groupby_date:
            xr_datacube = xr_datacube.groupby("time.date", restore_coord_dims=True)
            xr_datacube = getattr(xr_datacube, groupby_date)().rename(dict(date="time"))
            xr_datacube["time"] = xr_datacube.time.astype("M8[ns]")

        # To filter by cloud_cover / clear_cover, we need to compute clear pixels as field level
        if clear_cover or mask_statistics:
            xy = xr_datacube[mask_with].isel(time=0).size

            null_pixels = xr_datacube[mask_with].isnull().sum(dim=("x", "y"))
            n_pixels_as_labels = xy - null_pixels

            xr_datacube = xr_datacube.assign_coords({"clear_pixels": ("time", n_pixels_as_labels.data)})

            xr_datacube = xr_datacube.assign_coords(
                {
                    "clear_percent": (
                        "time",
                        np.multiply(
                            np.divide(
                                xr_datacube["clear_pixels"].data,
                                xr_datacube.attrs["usable_pixels"],
                            ),
                            100,
                        ).astype(np.int8),
                    )
                }
            )

            xr_datacube["clear_pixels"] = xr_datacube["clear_pixels"].load()
            xr_datacube["clear_percent"] = xr_datacube["clear_percent"].load()
        if mask_with:
            xr_datacube = xr_datacube.drop(mask_with)
        if clear_cover:
            xr_datacube = filter_clear_cover(xr_datacube, clear_cover)

        return xr_datacube

    def _update_search_for_assets(self, assets):
        fields = {
            "include": [
                "id",
                "type",
                "collection",
                "stac_version",
                "stac_extensions",
                "collection",
                "geometry",
                "bbox",
                "properties",
            ]
        }
        fields["include"].extend([f"assets.{asset}" for asset in assets])
        return fields

    @parallel_search
    def search(
        self,
        collections: str | list,
        intersects: gpd.GeoDataFrame = None,
        bbox=None,
        post_query=None,
        prefer_alternate=None,
        add_default_scale_factor=False,
        assets=None,
        raise_no_items=True,
        batch_days="auto",
        n_jobs=-1,
        **kwargs,
    ):
        """
        A wrapper around the pystac client search method. Add some features to enhance experience.

        Parameters
        ----------
        collections : str | list
            Collection(s) to search. It is recommended to only search one collection at a time.
        intersects : gpd.GeoDataFrame, optional
            If provided, the results will contain only intersecting items. The default is None.
        bbox : TYPE, optional
            If provided, the results will contain only intersecting items. The default is None.
        post_query : TYPE, optional
            STAC-like filters applied on retrieved items. The default is None.
        prefer_alternate : TYPE, optional
            Prefer alternate links when available. The default is None.
        **kwargs : TYPE
            Keyword arguments passed to the pystac client search method.

        Returns
        -------
        items_collection : ItemCollection
            The filtered STAC items.

        Example
        -------
        >>> items = eds.search(collections='sentinel-2-l2a',bbox=[1,43,1,43],datetime='2017')
        >>> len(items)
        27
        >>> print(items[0].id)
        S2A_31TCH_20170126_0_L2A
        >>> print(items[0].assets.keys())
        dict_keys(['aot', 'nir', 'red', 'scl', 'wvp', 'blue', 'green', 'nir08', 'nir09',
                   'swir16', 'swir22', 'visual', 'aot-jp2', 'coastal', 'nir-jp2',
                   'red-jp2', 'scl-jp2', 'wvp-jp2', 'blue-jp2', 'rededge1', 'rededge2',
                   'rededge3', 'green-jp2', 'nir08-jp2', 'nir09-jp2', 'thumbnail',
                   'swir16-jp2', 'swir22-jp2', 'visual-jp2', 'coastal-jp2',
                   'rededge1-jp2', 'rededge2-jp2', 'rededge3-jp2', 'granule_metadata',
                   'tileinfo_metadata'])
        >>> print(items[0].properties)
        {
            "created": "2020-09-01T04:59:33.629000Z",
            "updated": "2022-11-08T13:08:57.661605Z",
            "platform": "sentinel-2a",
            "grid:code": "MGRS-31TCH",
            "proj:epsg": 32631,
            "instruments": ["msi"],
            "s2:sequence": "0",
            "constellation": "sentinel-2",
            "mgrs:utm_zone": 31,
            "s2:granule_id": "S2A_OPER_MSI_L2A_TL_SHIT_20190506T054613_A008342_T31TCH_N00.01",
            "eo:cloud_cover": 26.518754,
            "s2:datatake_id": "GS2A_20170126T105321_008342_N00.01",
            "s2:product_uri": "S2A_MSIL2A_20170126T105321_N0001_R051_T31TCH_20190506T054611.SAFE",
            "s2:datastrip_id": "S2A_OPER_MSI_L2A_DS_SHIT_20190506T054613_S20170126T105612_N00.01",
            "s2:product_type": "S2MSI2A",
            "mgrs:grid_square": "CH",
            "s2:datatake_type": "INS-NOBS",
            "view:sun_azimuth": 161.807489888479,
            "eda:geometry_tags": ["RESOLVED_CLOCKWISE_POLYGON"],
            "mgrs:latitude_band": "T",
            "s2:generation_time": "2019-05-06T05:46:11.879Z",
            "view:sun_elevation": 26.561907592092602,
            "earthsearch:s3_path": "s3://sentinel-cogs/sentinel-s2-l2a-cogs/31/T/CH/2017/1/S2A_31TCH_20170126_0_L2A",
            "processing:software": {"sentinel2-to-stac": "0.1.0"},
            "s2:water_percentage": 0.697285,
            "eda:original_geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [0.5332306381710475, 43.32623760511659],
                        [1.887065663431107, 43.347431265475954],
                        [1.9046784554725638, 42.35884880571144],
                        [0.5722310999779479, 42.3383710796791],
                        [0.5332306381710475, 43.32623760511659],
                    ]
                ],
            },
            "earthsearch:payload_id": "roda-sentinel2/workflow-sentinel2-to-stac/80f56ba6349cf8e21c1424491f1589c2",
            "s2:processing_baseline": "00.01",
            "s2:snow_ice_percentage": 23.041981,
            "s2:vegetation_percentage": 15.52531,
            "s2:thin_cirrus_percentage": 0.563798,
            "s2:cloud_shadow_percentage": 4.039595,
            "s2:nodata_pixel_percentage": 0.000723,
            "s2:unclassified_percentage": 9.891956,
            "s2:dark_features_percentage": 15.112966,
            "s2:not_vegetated_percentage": 5.172154,
            "earthsearch:boa_offset_applied": False,
            "s2:degraded_msi_data_percentage": 0.0,
            "s2:high_proba_clouds_percentage": 18.044451,
            "s2:reflectance_conversion_factor": 1.03230935243016,
            "s2:medium_proba_clouds_percentage": 7.910506,
            "s2:saturated_defective_pixel_percentage": 0.0,
            "eda:tracking_id": "eZbRVxsbEGdWLKXDK2i9Ve",
            "eda:status": "PUBLISHED",
            "datetime": "2017-01-26T10:56:12.238000Z",
            "eda:loose_validation_status": "VALID",
            "eda:ag_cloud_mask_available": False,
        }

        """

        # Find available assets for a collection
        # And query only these assets to avoid requesting unused data
        if isinstance(collections, str):
            collections = [collections]
        if assets is not None:
            assets = list(AssetMapper().map_collection_assets(collections[0], assets).keys())
            kwargs["fields"] = self._update_search_for_assets(assets)

        if bbox is None and intersects is not None:
            intersects = GeometryManager(intersects).to_intersects(crs="4326")
        if bbox is not None and intersects is not None:
            bbox = None

        items_collection = self.client.search(
            collections=collections,
            bbox=bbox,
            intersects=intersects,
            sortby="properties.datetime",
            **kwargs,
        )

        # Downloading the items
        items_collection = items_collection.item_collection()

        # prefer_alternate means to prefer alternate url (to replace default href)
        if any((prefer_alternate, add_default_scale_factor)):
            items_collection = enhance_assets(
                items_collection.clone(),
                alternate=prefer_alternate,
                add_default_scale_factor=add_default_scale_factor,
            )
        if post_query:
            items_collection = post_query_items(items_collection, post_query)
        if len(items_collection) == 0 and raise_no_items:
            raise NoItemsFoundError("No items found.")
        return items_collection

    def find_cloud_mask_items(self, items_collection, cloudmask="ag_cloud_mask", **kwargs):
        """
        Search the catalog for the ag_cloud_mask items matching the given items_collection.
        The ag_cloud_mask items are searched in the `ag_cloud_mask_collection_id` collection using the
        `ag_cloud_mask_item_id` properties of the items.

        Parameters
        ----------
        items_collection : pystac.ItemCollection
            The items to find corresponding ag cloud mask items for.

        Returns
        -------
        pystac.ItemCollection
            The filtered item collection.
        """

        def ag_cloud_mask_from_items(items):
            products = {}
            for item in items:
                if not item.properties.get(f"eda:{cloudmask}_available"):
                    continue
                collection = item.properties[f"eda:{cloudmask}_collection_id"]
                if products.get(collection, None) is None:
                    products[collection] = []
                products[collection].append(item.properties.get(f"eda:{cloudmask}_item_id"))
            return products

        items_id = ag_cloud_mask_from_items(items_collection)
        if len(items_id) == 0:
            raise ValueError("Sorry, no ag_cloud_mask available.")
        collections = list(items_id.keys())
        ids_ = [x for n in (items_id.values()) for x in n]
        items_list = []
        step = 100
        kwargs.setdefault("prefer_alternate", "download")

        for items_start_idx in range(0, len(ids_), step):
            items = self.search(
                collections=collections,
                ids=ids_[items_start_idx : items_start_idx + step],
                limit=step,
                **kwargs,
            )
            items_list.extend(list(items))
        return ItemCollection(items_list)
