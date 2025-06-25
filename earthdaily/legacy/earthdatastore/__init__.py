import logging
import operator
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import xarray as xr
from odc import stac
from pystac.item_collection import ItemCollection
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

from earthdaily._api_requester import APIRequester

from . import _scales_collections, cube_utils, mask
from .cube_utils import _datacubes, asset_mapper, datacube, metacube
from .parallel_search import NoItemsFoundError, parallel_search

__all__ = ["datacube", "metacube", "xr", "stac"]

logging.getLogger("earthdaily-earthdatastore")


def apply_single_condition(item_value, condition_op: str, condition_value: Any | list[Any]) -> bool:
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


def validate_property_condition(item: Any, property_name: str, conditions: dict[str, Any]) -> bool:
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


def filter_items(items: ItemCollection | list[Any], query: dict[str, dict[str, Any]]) -> list[Any]:
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


def post_query_items(items: ItemCollection | list[Any], query: dict[str, dict[str, Any]]) -> ItemCollection:
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
                    scale_factor_collection = _scales_collections.scale_factor_collections.get(
                        item.collection_id if item.collection_id else "", [{}]
                    )
                    for scales_collection in scale_factor_collection:
                        assets_list = scales_collection.get("assets", [])
                        if isinstance(assets_list, list) and asset in assets_list:
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


class Auth:
    def __init__(
        self,
        api_requester: APIRequester,
        presign_urls=True,
        asset_proxy_enabled=False,
    ):
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

        self.api_requester = api_requester
        self.__presign_urls = presign_urls
        self.__asset_proxy_enabled = asset_proxy_enabled
        self._first_items_: dict = {}
        self._staccollectionexplorer: dict = {}
        self.client = self._get_client()

    def _get_client(self):
        """Get client for interacting with the EarthDataStore API.

        Returns
        -------
        client : Client
            A PySTAC client for interacting with the Earth Data Store STAC API.

        """

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.api_requester.headers,
            "Authorization": f"Bearer {self.api_requester.auth.get_token()}" if self.api_requester.auth else None,
            "X-Signed-Asset-Urls": str(self.__presign_urls),
            "X-Proxy-Asset-Urls": str(self.__asset_proxy_enabled),
        }

        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=None,
        )
        stac_api_io = StacApiIO(max_retries=retry)

        return Client.open(f"{self.api_requester.base_url}/platform/v1/stac", headers=headers, stac_io=stac_api_io)

    def explore(self, collection: Optional[str] = None):
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
         ...
        }
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
        intersects: gpd.GeoDataFrame | str | dict | None = None,
        bbox=None,
        mask_with: None | str | list = None,
        mask_statistics: bool | int = False,
        clear_cover: int | float | None = None,
        prefer_alternate: str | bool = "download",
        search_kwargs: dict = {},
        add_default_scale_factor: bool = True,
        common_band_names=True,
        cross_calibration_collection: None | str = None,
        properties: bool | str | list = False,
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
        ds : TYPE
            DESCRIPTION.

        """

        # Properties (per items) are not compatible with groupby_date.
        if properties not in (None, False) and groupby_date is not None:
            raise NotImplementedError("You must set `groupby_date=None` to have properties per item.")

        # convert collections to list
        collections = [collections] if isinstance(collections, str) else collections

        # if intersects a geometry, create a GeoDataFRame
        if intersects is not None:
            intersects = cube_utils.GeometryManager(intersects).to_geopandas()
            self.intersects = intersects

        # if mask_with, need to add assets or to get mask item id
        if mask_with:
            if mask_with not in mask._available_masks:
                raise NotImplementedError(
                    f"Specified mask '{mask_with}' is not available. \
                        Available masks providers are : {mask._available_masks}"
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
                mask_with = mask._native_mask_def_mapping.get(collections[0], None)
                sensor_mask = mask._native_mask_asset_mapping.get(collections[0], None)

                if isinstance(assets, list) and sensor_mask not in assets:
                    assets.append(sensor_mask)
                elif isinstance(assets, dict):
                    assets[sensor_mask] = sensor_mask
        bbox_query = None

        if bbox is None and intersects is not None:
            bbox_query = list(cube_utils.GeometryManager(intersects).to_bbox())
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
                    deduplicate_items=False,
                )
            except Warning:
                raise Warning("No cross calibration coefficient available for the specified collections.")

        # Create datacube from items
        ds = datacube(
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
            ds = ds.ed.clip(intersects)
        # Create mask datacube and apply it to ds
        if mask_with:
            if "geobox" not in kwargs:
                kwargs["geobox"] = ds.odc.geobox
            kwargs.pop("crs", "")
            kwargs.pop("resolution", "")
            kwargs["dtype"] = "int8"
            if clear_cover and mask_statistics is False:
                mask_statistics = True
            mask_kwargs = dict(mask_statistics=False)
            if mask_with == "ag_cloud_mask" or mask_with == "cloud_mask":
                mask_asset_mapping: dict[str, dict[str, str]] = {
                    "ag_cloud_mask": {"agriculture-cloud-mask": "ag_cloud_mask"},
                    "cloud_mask": {"cloud-mask": "cloud_mask"},
                }
                items_mask = self.find_cloud_mask_items(items, cloudmask=mask_with, **cloud_search_kwargs)
                ds_mask = datacube(
                    items_mask,
                    intersects=intersects,
                    bbox=bbox,
                    groupby_date=None,
                    assets=mask_asset_mapping[mask_with] if isinstance(mask_with, str) else None,
                    **kwargs,
                )
                ds["time"] = ds.time.astype("M8[ns]")
                if ds.time.size != ds_mask.time.size:
                    raise NotImplementedError(
                        "Sensor and cloudmask don't havethe same time.\
                                              {ds.time.size} for sensor and {ds_mask.time.size} for cloudmask."
                    )
                ds_mask["time"] = ds["time"].time
                ds_mask = cube_utils._match_xy_dims(ds_mask, ds)
                ds = xr.merge((ds, ds_mask), compat="override")

                # mask_kwargs.update(ds_mask=ds_mask)
            else:
                assets_mask = {
                    mask._native_mask_asset_mapping[collections[0]]: mask._native_mask_def_mapping[collections[0]]
                }

                ds_mask = datacube(
                    items,
                    groupby_date=None,
                    bbox=list(cube_utils.GeometryManager(intersects).to_bbox()),
                    assets=assets_mask,
                    resampling=0,
                    **kwargs,
                )
                ds_mask = cube_utils._match_xy_dims(ds_mask, ds)
                if intersects is not None:
                    ds_mask = ds_mask.ed.clip(intersects)
                ds = xr.merge((ds, ds_mask), compat="override")

            Mask = mask.Mask(ds, intersects=intersects, bbox=bbox)

            if not isinstance(mask_with, str):
                raise TypeError("mask_with must be a string")
            ds = getattr(Mask, mask_with)(**mask_kwargs)

        # keep only one value per pixel per day
        if groupby_date:
            ds = cube_utils._groupby_date(ds, groupby_date)

        # To filter by cloud_cover / clear_cover, we need to compute clear pixels as field level
        if clear_cover or mask_statistics:
            xy = ds[mask_with].isel(time=0).size

            null_pixels = ds[mask_with].isnull().sum(dim=("x", "y"))
            n_pixels_as_labels = xy - null_pixels

            ds = ds.assign_coords({"clear_pixels": ("time", n_pixels_as_labels.data)})

            ds = ds.assign_coords(
                {
                    "clear_percent": (
                        "time",
                        np.multiply(
                            np.divide(
                                ds["clear_pixels"].data,
                                ds.attrs["usable_pixels"],
                            ),
                            100,
                        ).astype(np.int8),
                    )
                }
            )

            ds["clear_pixels"] = ds["clear_pixels"].load()
            ds["clear_percent"] = ds["clear_percent"].load()
        if mask_with:
            ds = ds.drop_vars(mask_with)
        if clear_cover:
            ds = mask.filter_clear_cover(ds, clear_cover)

        return ds

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
        deduplicate_items=True,
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
        dict_keys(['aot', 'nir', 'red', ... , 'tileinfo_metadata'])
        >>> print(items[0].properties)
        {
            "created": "2020-09-01T04:59:33.629000Z",
            "updated": "2022-11-08T13:08:57.661605Z",
            "platform": "sentinel-2a",
            "grid:code": "MGRS-31TCH",
            ...
        }

        """

        # Find available assets for a collection
        # And query only these assets to avoid requesting unused data
        if isinstance(collections, str):
            collections = [collections]
        if assets is not None:
            assets = list(asset_mapper.AssetMapper().map_collection_assets(collections[0], assets).keys())
            kwargs["fields"] = self._update_search_for_assets(assets)

        if bbox is None and intersects is not None:
            intersects = cube_utils.GeometryManager(intersects).to_intersects(crs="4326")
        if bbox is not None and intersects is not None:
            bbox = None

        items_search = self.client.search(
            collections=collections,
            bbox=bbox,
            intersects=intersects,
            sortby="properties.datetime",
            **kwargs,
        )

        # Downloading the items
        items_collection = items_search.item_collection()

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
        if deduplicate_items:
            from ._filter_duplicate import filter_duplicate_items

            items_collection = filter_duplicate_items(items_collection)
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
                deduplicate_items=False,
                **kwargs,
            )
            items_list.extend(list(items))
        return ItemCollection(items_list)
