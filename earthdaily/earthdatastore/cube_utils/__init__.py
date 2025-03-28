# mypy: ignore-errors
# TODO (v1): Fix type issues and remove 'mypy: ignore-errors' after verifying non-breaking changes

import json
import logging
import warnings
from collections import defaultdict
from functools import wraps
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from shapely.geometry import box

from earthdaily.core import options

from ._zonal import zonal_stats
from .asset_mapper import AssetMapper
from .geometry_manager import GeometryManager
from .harmonizer import Harmonizer

logging.getLogger("earthdaily-cube_utils")

__all__ = ["GeometryManager", "rioxarray", "zonal_stats"]
_auto_mask_order = ["cloudmask", "ag_cloud_mask", "native"]


def _groupby_date(ds, func):
    if ds.time.size != np.unique(ds.time.dt.date).size:
        kwargs = {}
        if xr.get_options()["use_flox"]:
            kwargs = dict(engine=options.groupby_date_engine, skipna=True)
        ds = ds.groupby("time.date")
        ds = getattr(ds, func)(**kwargs).rename(dict(date="time"))
        ds["time"] = ds.time.astype("M8[ns]")
    return ds


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
    if "latitude" in ds.coords and "longitude" in ds.coords:
        ds = ds.rename({"latitude": "y", "longitude": "x"})
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


def _disable_known_datacube_warning():
    if options.disable_known_warning:
        warnings.filterwarnings(
            "ignore",
            message="Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.",
            category=NotGeoreferencedWarning,
            module="rasterio.warp",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in cast",
            module="dask.array.chunk",
        )
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="All-NaN slice encountered"
        )
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Mean of empty slice"
        )


def datacube(
    items_collection=None,
    bbox=None,
    intersects=None,
    assets: list | dict | None = None,
    engine="odc",
    rescale=True,
    groupby_date="mean",
    common_band_names=True,
    cross_calibration_items: list | None = None,
    properties: bool | str | list = False,
    **kwargs,
):
    _disable_known_datacube_warning()
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
            empty_dict_list: list = []
            band_idx = 1
            asset = ds_asset
            if len(parts := ds_asset.split(".")) == 2:
                index = parts[1][-1]
                is_band = isinstance(index, int) or (
                    isinstance(index, str) and index.isdigit()
                )
                if is_band:
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
            break
    # apply nodata
    ds = _apply_nodata(ds, nodatas)
    if rescale:
        ds = rescale_assets_with_items(items_collection, ds, assets=assets)

    # drop na dates
    ds = ds.isel(dict(time=np.where(~np.isnan(ds.time))[0]))

    if groupby_date:
        ds = _groupby_date(ds, groupby_date)
    if bbox is not None and intersects is None:
        intersects = _bbox_to_intersects(bbox)
    if intersects is not None:
        intersects = GeometryManager(intersects).to_geopandas()

    if isinstance(intersects, gpd.GeoDataFrame):
        # optimize by perclipping using bbox
        # no need anymore thanks to geobox/geopolygon in doc
        # ds = ds.rio.clip_box(*intersects.to_crs(ds.rio.crs).total_bounds)
        ds = ds.rio.clip(intersects.to_crs(ds.rio.crs).geometry)
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
    ds = ds.sortby(ds.time)
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
    unique_items: dict = {}
    for item in items_collection:
        unique_items.setdefault(item.datetime, item)

    items_collection_per_date = list(unique_items.values())

    # Validate items match dataset time
    if (
        len(items_collection_per_date) != ds.time.size
        and len(items_collection) != ds.time.size
    ):
        raise ValueError(
            "Mismatch between items and datacube time. Set rescale to False."
        )

    # Prepare assets list
    assets = assets or list(ds.data_vars.keys())
    scales: dict[str, dict[float, dict[float, list]]] = {}

    # Process scaling for each time step
    for idx, time in enumerate(ds.time.values):
        item = items_collection[idx]

        # Date validation
        if pd.Timestamp(time).strftime("%Y%m%d") != item.datetime.strftime("%Y%m%d"):
            raise ValueError(
                "Mismatch between items and datacube dates. Set rescale to False."
            )

        # BOA offset handling for Sentinel-2 L2A
        boa_offset_applied = item.properties.get(
            "earthsearch:boa_offset_applied", False
        )
        if boa_offset_applied_control and item.collection_id == "sentinel-2-l2a":
            if boa_offset_applied_force_by_date:
                boa_offset_applied = pd.Timestamp(time) >= pd.Timestamp("2022-02-28")

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
            scales.setdefault(ds_asset, {}).setdefault(scale, {}).setdefault(
                offset, []
            ).append(idx)

    # Apply rescaling
    if scales:
        scaled_assets = []
        for asset, scale_data in scales.items():
            asset_scaled = []
            for scale, offset_data in scale_data.items():
                for offset, times in offset_data.items():
                    asset_scaled.append(ds[[asset]].isel(time=times) * scale + offset)
            scaled_assets.append(xr.concat(asset_scaled, dim="time").sortby("time"))
        # Merge scaled assets
        ds_scaled = xr.merge(scaled_assets, join="override", compat="override").sortby(
            "time"
        )

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
                cubes[idx][data_var] = cubes[idx][data_var].where(
                    cubes[idx][data_var] == np.nan, other=np.nan
                )
    cube = xr.concat([_drop_unfrozen_coords(cube) for cube in cubes], dim=concat_dim)
    cube = _groupby(cube, by=by, how=how)
    cube = cube.sortby(cube.time)
    return _propagade_rio(cubes[0], cube)
