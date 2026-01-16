import logging
from typing import Any, Callable, Sequence

import pandas as pd
import xarray as xr
from odc import stac
from pystac import Item
from rasterio.enums import Resampling

from earthdaily.datacube._geometry import bbox_to_geopandas, geometry_to_geopandas
from earthdaily.datacube.constants import (
    DEFAULT_BBOX_CRS,
    DEFAULT_CHUNKS,
    DEFAULT_DTYPE,
    DEFAULT_ENGINE,
    DEFAULT_HREF_PATH,
    DEFAULT_NODATA,
    DIM_LATITUDE,
    DIM_LONGITUDE,
    DIM_TIME,
    DIM_X,
    DIM_Y,
)
from earthdaily.datacube.exceptions import DatacubeCreationError

logger = logging.getLogger(__name__)

EngineLoader = Callable[..., xr.Dataset]
_ENGINE_LOADERS: dict[str, EngineLoader] = {}


def register_engine_loader(engine: str, loader: EngineLoader) -> None:
    """
    Register a custom datacube engine loader.

    This function allows extending the datacube module with additional engines
    (e.g., stackstac) beyond the default 'odc' engine. The loader function must
    accept the same parameters as `_load_datacube_with_odc` and return an xarray Dataset.

    Parameters
    ----------
    engine : str
        Engine identifier (e.g., 'stackstac', 'custom')
    loader : EngineLoader
        Callable that takes STAC items and returns an xarray Dataset

    Example
    -------
    To add stackstac support:
        from earthdaily.datacube._builder import register_engine_loader
        def _load_with_stackstac(items, **kwargs):
            import stackstac
            return stackstac.stack(items, **kwargs)
        register_engine_loader('stackstac', _load_with_stackstac)
    """
    _ENGINE_LOADERS[engine] = loader


def unregister_engine_loader(engine: str) -> None:
    """
    Unregister a datacube engine loader.

    Parameters
    ----------
    engine : str
        Engine identifier to remove
    """
    _ENGINE_LOADERS.pop(engine, None)


def _replace_item_hrefs(items: list[Item], href_path: str) -> list[Item]:
    if not href_path:
        return items

    for item in items:
        for asset in item.assets.values():
            href_dict: dict[str, Any] = asset.to_dict()
            for key in href_path.split("."):
                href_dict = href_dict.get(key, {})
                if not isinstance(href_dict, dict):
                    break
            if isinstance(href_dict, str) and href_dict:
                asset.href = href_dict
    return items


def _get_dedup_key_value(item: Item, key: str) -> str:
    """Extract a value from a STAC item for deduplication key building."""
    if key == "date":
        if item.datetime:
            return item.datetime.strftime("%Y-%m-%d")
        return item.id
    elif key == "collection_id":
        return item.collection_id or ""
    else:
        return str(item.properties.get(key, ""))


def _deduplicate_items(
    items: Sequence[Item],
    deduplicate_by: list[str],
    deduplicate_keep: str = "last",
) -> list[Item]:
    """
    Deduplicate STAC items based on specified properties.

    Parameters
    ----------
    items : Sequence[Item]
        List of STAC items to deduplicate
    deduplicate_by : list[str]
        List of properties to use as composite key for deduplication.
        Supported values:
        - "date": Date part of item.datetime (YYYY-MM-DD)
        - "collection_id": item.collection_id
        - Any other string: Looks up in item.properties (e.g., "proj:transform")
    deduplicate_keep : str
        Which item to keep when duplicates found: "first" or "last".
        Default is "last".

    Returns
    -------
    list[Item]
        Deduplicated items
    """
    if deduplicate_keep not in ("first", "last"):
        raise ValueError(f"deduplicate_keep must be 'first' or 'last', got '{deduplicate_keep}'")

    seen: dict[tuple[str, ...], Item] = {}
    for item in items:
        key = tuple(_get_dedup_key_value(item, k) for k in deduplicate_by)

        if deduplicate_keep == "last":
            seen[key] = item
        else:
            if key not in seen:
                seen[key] = item

    original_count = len(items)
    deduped = list(seen.values())
    if len(deduped) < original_count:
        logger.info(
            f"Deduplicated items: {original_count} -> {len(deduped)} (by {deduplicate_by}, keep={deduplicate_keep})"
        )

    return deduped


def build_datacube(
    items: Sequence[Item],
    assets: list[str] | dict[str, str] | None = None,
    bbox: list[float] | tuple | None = None,
    intersects: Any = None,
    dtype: str = DEFAULT_DTYPE,
    nodata: float | int | None = DEFAULT_NODATA,
    properties: bool | str | list[str] = False,
    engine: str = DEFAULT_ENGINE,
    replace_href_with: str = DEFAULT_HREF_PATH,
    **kwargs,
) -> xr.Dataset:
    """
    Build an xarray Dataset from STAC items using the specified engine.

    The engine system is extensible - additional engines (e.g., stackstac) can be
    registered using `register_engine_loader()`. Currently, only 'odc' is registered
    by default, but the architecture supports future engine additions.

    Parameters
    ----------
    items : list[Item]
        List of PySTAC Item objects
    assets : list[str] | dict[str, str] | None
        Assets to load. If dict, maps original names to desired output names.
    bbox : list[float] | tuple | None
        Bounding box [minx, miny, maxx, maxy] for spatial filtering
    intersects : Any
        Geometry for spatial intersection (GeoJSON, WKT, GeoDataFrame)
    dtype : str
        Data type for loaded arrays (default: 'float32')
    nodata : float | int | None
        Fill value to use for missing pixels. Passed to the engine loader
        (and ultimately odc.stac.load) so users can pick values such as
        NaN or -9999. Default uses the engine default.
    properties : bool | str | list[str]
        STAC properties to include as coordinates
    engine : str
        Engine to use for datacube creation. Default: 'odc'.
        Additional engines can be registered via `register_engine_loader()`.
    replace_href_with : str
        Path to alternate href in asset dictionary (dot notation, e.g., 'alternate.download.href')
    **kwargs
        Additional arguments passed to the engine loader (odc.stac.load).
        Common options include:

        - **pool** : int, optional
            Thread pool size for parallel COG metadata fetching.
            Recommended: `os.cpu_count()` for faster datacube creation.
            Example: `datacube.create(..., pool=8)`
        - **resolution** : float, optional
            Output resolution in CRS units
        - **crs** : str, optional
            Target coordinate reference system (e.g., 'EPSG:4326')
        - **resampling** : str, optional
            Resampling method (e.g., 'nearest', 'bilinear')
        - **fail_on_error** : bool, optional
            If True (default), raise an error when an asset fails to load (e.g., 404).
            If False, skip failed assets and continue loading. Useful when some
            assets may be missing or corrupted.

    Returns
    -------
    xr.Dataset
        Xarray Dataset with dimensions (time, y, x) and data variables for each asset

    Raises
    ------
    DatacubeCreationError
        If the engine is not supported or datacube creation fails
    """
    loader = _ENGINE_LOADERS.get(engine)
    if loader is None:
        available = ", ".join(sorted(_ENGINE_LOADERS.keys())) or "none"
        raise DatacubeCreationError(f"Engine '{engine}' not supported. Available engines: {available}")

    logger.info(f"Building datacube with {len(items)} items using {engine} engine")
    return loader(
        items=items,
        assets=assets,
        bbox=bbox,
        intersects=intersects,
        dtype=dtype,
        nodata=nodata,
        properties=properties,
        replace_href_with=replace_href_with,
        **kwargs,
    )


def _load_datacube_with_odc(
    *,
    items: Sequence[Item],
    assets: list[str] | dict[str, str] | None = None,
    bbox: list[float] | tuple | None = None,
    intersects: Any = None,
    dtype: str = DEFAULT_DTYPE,
    nodata: float | int | None = DEFAULT_NODATA,
    properties: bool | str | list[str] = False,
    replace_href_with: str = DEFAULT_HREF_PATH,
    **kwargs,
) -> xr.Dataset:
    if not items:
        raise DatacubeCreationError("No items provided to build datacube")

    if assets:
        available_assets = set(items[0].assets.keys())
        requested_assets = list(assets.keys()) if isinstance(assets, dict) else assets
        missing_assets = [asset for asset in requested_assets if asset not in available_assets]

        if missing_assets:
            raise DatacubeCreationError(
                f"Requested assets not found: {missing_assets}. Available assets: {sorted(available_assets)}"
            )

    if replace_href_with:
        items = _replace_item_hrefs(list(items), replace_href_with)

    if "epsg" in kwargs:
        kwargs["crs"] = f"EPSG:{kwargs['epsg']}"
        kwargs.pop("epsg")

    if "resampling" in kwargs:
        if isinstance(kwargs["resampling"], int):
            kwargs["resampling"] = Resampling(kwargs["resampling"]).name

    kwargs["chunks"] = kwargs.get("chunks", DEFAULT_CHUNKS)

    if nodata is not None:
        if "nodata" in kwargs and kwargs["nodata"] != nodata:
            logger.warning("Overriding nodata value from kwargs with explicit nodata argument.")
        kwargs["nodata"] = nodata

    if "geobox" in kwargs and "geopolygon" in kwargs:
        kwargs.pop("geopolygon")

    if intersects is not None:
        kwargs["geopolygon"] = geometry_to_geopandas(intersects)
    elif bbox is not None:
        bbox_crs = kwargs.get("crs", DEFAULT_BBOX_CRS)
        kwargs["geopolygon"] = bbox_to_geopandas(bbox, crs=bbox_crs)

    assets_keys = None
    if isinstance(assets, dict):
        assets_keys = list(assets.keys())

    try:
        ds = stac.load(
            items,
            bands=assets_keys if isinstance(assets, dict) else assets,
            preserve_original_order=True,
            dtype=dtype,
            groupby=None,
            **kwargs,
        )
    except ValueError as e:
        if "No such band/alias" in str(e):
            available_assets_list = sorted(items[0].assets.keys())
            raise DatacubeCreationError(
                f"Asset not found in STAC items. Error: {str(e)}. Available assets: {available_assets_list}"
            ) from e
        raise DatacubeCreationError(f"Failed to create datacube: {str(e)}") from e
    except Exception as e:
        raise DatacubeCreationError(f"Unexpected error creating datacube: {str(e)}") from e

    if properties:
        if DIM_TIME not in ds.coords:
            logger.debug(f"Skipping metadata assignment because dataset has no '{DIM_TIME}' coordinate.")
        else:
            if properties is True:
                selected_properties = sorted({k for item in items for k in item.properties.keys()})
            elif isinstance(properties, str):
                selected_properties = [properties]
            elif isinstance(properties, list):
                selected_properties = list(properties)
            else:
                selected_properties = []

            metadata_rows: list[dict[str, Any]] = []
            missing_time_items = 0

            for item in items:
                timestamp = getattr(item, "datetime", None) or item.properties.get("datetime")
                if timestamp is None:
                    missing_time_items += 1
                    continue

                row: dict[str, Any] = {"__time__": timestamp}
                for key in selected_properties:
                    value = item.properties.get(key)
                    if isinstance(value, list):
                        value = str(value)
                    row[key] = value

                metadata_rows.append(row)

            if metadata_rows:
                df = pd.DataFrame(metadata_rows)
                metadata_times = df["__time__"].tolist()
                metadata_index = pd.to_datetime(metadata_times, utc=True).tz_convert(None)
                df = df.drop(columns="__time__")
                df.index = metadata_index
                df = df[~df.index.duplicated(keep="last")]

                dataset_times = pd.to_datetime(ds.coords[DIM_TIME].values, utc=True).tz_convert(None)
                df = df.reindex(dataset_times)

                metadata_coords: dict[str, Any] = {k: (DIM_TIME, df[k].tolist()) for k in df.columns}
                if metadata_coords:
                    ds = ds.assign_coords(**metadata_coords)

            if missing_time_items:
                logger.debug(f"Skipped {missing_time_items} STAC item(s) without datetime when aligning metadata.")

    if DIM_LATITUDE in ds.coords and DIM_LONGITUDE in ds.coords:
        ds = _normalize_longitude_coordinate(ds)
        ds = ds.rename({DIM_LATITUDE: DIM_Y, DIM_LONGITUDE: DIM_X})

    ds = ds.chunk(kwargs["chunks"])

    if isinstance(assets, dict):
        ds = ds.rename_vars({k: v for k, v in assets.items() if k in ds.data_vars})

    return ds


register_engine_loader(DEFAULT_ENGINE, _load_datacube_with_odc)


def _normalize_longitude_coordinate(ds: xr.Dataset) -> xr.Dataset:
    """
    Remap 0-360 longitude coordinates to the conventional -180 to 180 range when detected.
    """
    if DIM_LONGITUDE not in ds.coords:
        return ds

    lon_coord = ds[DIM_LONGITUDE]
    try:
        lon_min = float(lon_coord.min().item())
        lon_max = float(lon_coord.max().item())
    except Exception:  # pragma: no cover
        return ds

    if lon_min >= 0 and lon_max > 180:
        new_longitudes = ((lon_coord + 180) % 360) - 180
        ds = ds.assign_coords({DIM_LONGITUDE: new_longitudes}).sortby(DIM_LONGITUDE)
        logger.debug("Remapped longitude coordinates from 0-360 to -180 to 180 range")

    return ds
