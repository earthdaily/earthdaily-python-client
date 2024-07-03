import json
import logging
import operator
import os
import warnings
import time
import pandas as pd
import geopandas as gpd
import psutil
import requests
import xarray as xr
import numpy as np
from pystac.item_collection import ItemCollection
from pystac_client.item_search import ItemSearch
from pystac_client import Client
from itertools import chain
from odc import stac
from . import _scales_collections, cube_utils, mask
from .cube_utils import datacube, metacube, _datacubes, asset_mapper

__all__ = ["datacube", "metacube", "xr", "stac"]

logging.getLogger("earthdaily-earthdatastore")


class NoItems(Warning):
    pass


_no_item_msg = NoItems("No item has been found for your query.")


def _datetime_to_str(datetime):
    start, end = ItemSearch(url=None)._format_datetime(datetime).split("/")
    return start, end


def _datetime_split(datetime, freq="auto"):
    start, end = [pd.Timestamp(date) for date in _datetime_to_str(datetime)]
    diff = end - start
    if freq == "auto":
        # freq increases of 5 days every 6 months
        freq = diff // (5 + 5 * (diff.days // 183))
    else:
        freq = pd.Timedelta(days=freq)
    if diff.days < freq.days:
        return datetime
    logging.info(f"Parallel search with datetime split every {freq.days} days.")
    return [
        (chunk, min(chunk + freq, end))
        for chunk in pd.date_range(start, end, freq=freq)[:-1]
    ]


def _parallel_search(func):
    def _search(*args, **kwargs):
        from joblib import Parallel, delayed

        kwargs.setdefault("batch_days", "auto")
        batch_days = kwargs.get("batch_days", None)
        datetime = kwargs.get("datetime", None)
        need_parallel = False
        if datetime and batch_days is not None:
            datetimes = _datetime_split(datetime, batch_days)
            need_parallel = True if len(datetimes) > 1 else False
            if need_parallel:
                kwargs.pop("datetime")
                kwargs["raise_no_items"] = False
                items = Parallel(n_jobs=10, backend="threading")(
                    delayed(func)(*args, datetime=datetime, **kwargs)
                    for datetime in datetimes
                )
                items = ItemCollection(chain(*items))
                if len(items) == 0:
                    raise _no_item_msg
        if not need_parallel:
            items = func(*args, **kwargs)
        return items

    return _search


def post_query_items(items, query):
    """Applies query to items fetched from the STAC catalog.

    Parameters
    ----------
    items : list
        List of items
    query : dict
        Query to post

    Returns
    -------
    items : ItemCollection
        filtered items
    """
    items_ = []
    for idx, item in enumerate(items):
        queries_results = 0
        for k, v in query.items():
            if k not in item.properties.keys():
                continue
            for v_op, v_val in v.items():
                if isinstance(v_val, list):
                    results = 0
                    for v_val_ in v_val:
                        operation = operator.__dict__[v_op](item.properties[k], v_val_)

                        if operation:
                            results += 1
                    if results == len(v_val):
                        queries_results += 1
                else:
                    operation = operator.__dict__[v_op](item.properties[k], v_val)
                    if operation:
                        queries_results += 1
        if queries_results == len(query.keys()):
            items_.append(item)

    items = ItemCollection(items_)
    return items


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
    items,
    alternate="download",
    use_http_url=False,
    add_default_scale_factor=False,
):
    """Enhance assets with extra fields.

    Parameters
    ----------
    items : ItemCollection
        A PySTAC ItemCollection
    alternate : str, optional
        Alternate asset to use, by default "download"
    use_http_url : bool, optional
        Use HTTP URL instead of cloud path, by default False
    add_default_scale_factor : bool, optional
        Add default scale, offset, nodata factor to assets, by default False

    Returns
    -------
    items : ItemCollection
        Updated PySTAC ItemCollection
    """

    if any((alternate, use_http_url, add_default_scale_factor)):
        for idx, item in enumerate(items):
            keys = list(item.assets.keys())
            for asset in keys:
                # use the alternate href if it exists
                if alternate:
                    href = (
                        item.assets[asset]
                        .extra_fields.get("alternate", {})
                        .get(alternate, {})
                        .get("href")
                    )
                    if href:
                        items[idx].assets[asset].href = href
                # use HTTP URL instead of cloud path
                if use_http_url:
                    href = item.assets[asset].to_dict().get("href", {})
                    if href:
                        items[idx].assets[asset].href = _cloud_path_to_http(href)
                if add_default_scale_factor:
                    scale_factor_collection = (
                        _scales_collections.scale_factor_collections.get(
                            item.collection_id, [{}]
                        )
                    )
                    for scales_collection in scale_factor_collection:
                        if asset in scales_collection.get("assets", []):
                            if (
                                "raster:bands"
                                not in items[idx].assets[asset].extra_fields
                            ):
                                items[idx].assets[asset].extra_fields[
                                    "raster:bands"
                                ] = [{}]
                            if (
                                not items[idx]
                                .assets[asset]
                                .extra_fields["raster:bands"][0]
                                .get("scale")
                            ):
                                items[idx].assets[asset].extra_fields["raster:bands"][
                                    0
                                ]["scale"] = scales_collection["scale"]
                                items[idx].assets[asset].extra_fields["raster:bands"][
                                    0
                                ]["offset"] = scales_collection["offset"]
                                items[idx].assets[asset].extra_fields["raster:bands"][
                                    0
                                ]["nodata"] = scales_collection["nodata"]

    return items


def _get_token(config=None, presign_urls=True):
    """Get token for interacting with the Earth Data Store API.

    By default, Earth Data Store will look for environment variables called
    EDS_AUTH_URL, EDS_SECRET and EDS_CLIENT_ID.

    Parameters
    ----------
    config : str | dict, optional
        A JSON string or a dictionary with the credentials for the Earth Data Store.
    presign_urls : bool, optional
        Use presigned URLs, by default True

    Returns
    -------
    token : str
    eds_url : the earthdatastore url

    """
    if config is None:
        config = os.getenv
    auth_url = config("EDS_AUTH_URL")
    secret = config("EDS_SECRET")
    client_id = config("EDS_CLIENT_ID")
    eds_url = config("EDS_API_URL", "https://api.eds.earthdaily.com/archive/v1/stac/v1")
    if auth_url is None or secret is None or client_id is None:
        raise AttributeError(
            "You need to have env : EDS_AUTH_URL, EDS_SECRET and EDS_CLIENT_ID"
        )

    token_response = requests.post(
        auth_url,
        data={"grant_type": "client_credentials"},
        allow_redirects=False,
        auth=(client_id, secret),
    )
    token_response.raise_for_status()
    return json.loads(token_response.text)["access_token"], eds_url


def _get_client(config=None, presign_urls=True):
    """Get client for interacting with the Earth Data Store API.

    By default, Earth Data Store will look for environment variables called
    EDS_AUTH_URL, EDS_SECRET and EDS_CLIENT_ID.

    Parameters
    ----------
    config : str | dict, optional
        A JSON string or a dictionary with the credentials for the Earth Data Store.
    presign_urls : bool, optional
        Use presigned URLs, by default True

    Returns
    -------
    client : Client
        A PySTAC client for interacting with the Earth Data Store STAC API.

    """

    if isinstance(config, tuple):  # token
        token, eds_url = config
        logging.log(level=logging.INFO, msg="Using token to reauth")
    else:
        if isinstance(config, dict):
            config = config.get
        elif isinstance(config, str) and config.endswith(".json"):
            config = json.load(open(config, "rb")).get
        token, eds_url = _get_token(config, presign_urls)

    headers = {"Authorization": f"bearer {token}"}
    if presign_urls:
        headers["X-Signed-Asset-Urls"] = "True"

    return Client.open(
        eds_url,
        headers=headers,
    )


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
    def __init__(self, config: str | dict = None, presign_urls=True):
        """
        A client for interacting with the Earth Data Store API.
        By default, Earth Data Store will look for environment variables called
        EDS_AUTH_URL, EDS_SECRET and EDS_CLIENT_ID.

        Parameters
        ----------
        config : str | dict, optional
            The path to the json file containing the Earth Data Store credentials,
            or a dict with those credentials.

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
        self._client = None
        self.__auth_config = config
        self.__presign_urls = presign_urls
        self._first_items_ = {}
        self._staccollectionexplorer = {}
        self.__time_eds_log = time.time()
        self._client = self.client

    @property
    def client(self):
        """
                Create an instance of pystac client from EarthDataSTore

                Returns
                -------
                catalog : A :class:`Client` instance for this Catalog.
        .

        """
        if t := (time.time() - self.__time_eds_log) > 3600 or self._client is None:
            if t:
                logging.log(level=logging.INFO, msg="Reauth to EarthDataStore")
            self._client = _get_client(self.__auth_config, self.__presign_urls)
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
                self._staccollectionexplorer[collection] = StacCollectionExplorer(
                    self.client, collection
                )
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
        if properties not in (None, False) and groupby_date is not None:
            raise NotImplementedError(
                "You must set `groupby_date=None` to have properties per item."
            )

        if isinstance(collections, str):
            collections = [collections]

        if intersects is not None:
            intersects = cube_utils.GeometryManager(intersects).to_geopandas()
            self.intersects = intersects
        if mask_with:
            if mask_with not in mask._available_masks:
                raise NotImplementedError(
                    f"Specified mask '{mask_with}' is not available. Available masks providers are : {mask._available_masks}"
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

        items = self.search(
            collections=collections,
            bbox=bbox,
            intersects=intersects,
            datetime=datetime,
            assets=assets,
            prefer_alternate=prefer_alternate,
            add_default_scale_factor=add_default_scale_factor,
            **search_kwargs,
        )

        xcal_items = None
        if (
            isinstance(cross_calibration_collection, str)
            and cross_calibration_collection != collections[0]
        ):
            try:
                xcal_items = self.search(
                    collections="eda-cross-calibration",
                    intersects=intersects,
                    query={
                        "eda_cross_cal:source_collection": {"eq": collections[0]},
                        "eda_cross_cal:destination_collection": {
                            "eq": cross_calibration_collection
                        },
                    },
                )
            except Warning:
                raise Warning(
                    "No cross calibration coefficient available for the specified collections."
                )
        kwargs.setdefault("dtype", "float32")
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
        if mask_with:
            kwargs["dtype"] = "int8"
            if "geobox" not in kwargs:
                kwargs["geobox"] = xr_datacube.odc.geobox

            if clear_cover and mask_statistics is False:
                mask_statistics = True
            mask_kwargs = dict(mask_statistics=False)
            if mask_with == "ag_cloud_mask" or mask_with == "cloud_mask":
                mask_asset_mapping = {
                    "ag_cloud_mask": {"agriculture-cloud-mask": "ag_cloud_mask"},
                    "cloud_mask": {"cloud-mask": "cloud_mask"},
                }
                acm_items = self.find_cloud_mask_items(items, cloudmask=mask_with)
                acm_datacube = datacube(
                    acm_items,
                    intersects=intersects,
                    bbox=bbox,
                    groupby_date=None,
                    assets=mask_asset_mapping[mask_with],
                    **kwargs,
                )
                xr_datacube["time"] = xr_datacube.time.astype("M8[ns]")
                acm_datacube["time"] = xr_datacube["time"].time
                acm_datacube = cube_utils._match_xy_dims(acm_datacube, xr_datacube)
                xr_datacube = xr.merge((xr_datacube, acm_datacube), compat="override")

                # mask_kwargs.update(acm_datacube=acm_datacube)
            else:
                mask_assets = {
                    mask._native_mask_asset_mapping[
                        collections[0]
                    ]: mask._native_mask_def_mapping[collections[0]]
                }

                if "resolution" in kwargs:
                    kwargs.pop("resolution")
                if "epsg" in kwargs:
                    kwargs.pop("epsg")

                clouds_datacube = datacube(
                    items,
                    groupby_date=None,
                    intersects=intersects,
                    bbox=bbox,
                    assets=mask_assets,
                    resampling=0,
                    **kwargs,
                )
                clouds_datacube = cube_utils._match_xy_dims(
                    clouds_datacube, xr_datacube
                )
                xr_datacube = xr.merge(
                    (xr_datacube, clouds_datacube), compat="override"
                )

            Mask = mask.Mask(xr_datacube, intersects=intersects, bbox=bbox)
            xr_datacube = getattr(Mask, mask_with)(**mask_kwargs)

        if groupby_date:
            xr_datacube = xr_datacube.groupby("time.date", restore_coord_dims=True)
            xr_datacube = getattr(xr_datacube, groupby_date)().rename(dict(date="time"))
            xr_datacube["time"] = xr_datacube.time.astype("M8[ns]")

        if clear_cover or mask_statistics:
            xy = xr_datacube[mask_with].isel(time=0).size

            null_pixels = xr_datacube[mask_with].isnull().sum(dim=("x", "y"))
            n_pixels_as_labels = xy - null_pixels

            xr_datacube = xr_datacube.assign_coords(
                {"clear_pixels": ("time", n_pixels_as_labels.load().values)}
            )

            xr_datacube = xr_datacube.assign_coords(
                {
                    "clear_percent": (
                        "time",
                        np.multiply(
                            xr_datacube["clear_pixels"].values
                            / xr_datacube.attrs["usable_pixels"],
                            100,
                        ).astype(np.int8),
                    )
                }
            )
        if mask_with:
            xr_datacube = xr_datacube.drop(mask_with)
        if clear_cover:
            xr_datacube = mask.filter_clear_cover(xr_datacube, clear_cover)

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

    @_parallel_search
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
        if assets is not None:
            assets = list(
                asset_mapper.AssetMapper()
                .map_collection_assets(collections[0], assets)
                .keys()
            )
            kwargs["fields"] = self._update_search_for_assets(assets)
        if isinstance(collections, str):
            collections = [collections]
        if bbox is None and intersects is not None:
            intersects = cube_utils.GeometryManager(intersects).to_intersects(
                crs="4326"
            )
        if bbox is not None and intersects is not None:
            bbox = None

        items_collection = self.client.search(
            collections=collections,
            bbox=bbox,
            intersects=intersects,
            sortby="properties.datetime",
            **kwargs,
        )
        items_collection = items_collection.item_collection()

        if any((prefer_alternate, add_default_scale_factor)):
            items_collection = enhance_assets(
                items_collection.clone(),
                alternate=prefer_alternate,
                add_default_scale_factor=add_default_scale_factor,
            )
        if post_query:
            items_collection = post_query_items(items_collection, post_query)
        if len(items_collection) == 0 and raise_no_items:
            raise _no_item_msg
        return items_collection

    def find_cloud_mask_items(self, items_collection, cloudmask="ag_cloud_mask"):
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
                products[collection].append(
                    item.properties.get(f"eda:{cloudmask}_item_id")
                )
            return products

        items_id = ag_cloud_mask_from_items(items_collection)
        if len(items_id) == 0:
            raise ValueError("Sorry, no ag_cloud_mask available.")
        collections = list(items_id.keys())
        ids_ = [x for n in (items_id.values()) for x in n]
        items_list = []
        step = 100
        for items_start_idx in range(0, len(ids_), step):
            items = self.search(
                collections=collections,
                # intersects=self.intersects,
                ids=ids_[items_start_idx : items_start_idx + step],
                limit=step,
            )
            items_list.extend(list(items))
        return ItemCollection(items_list)


def item_property_to_df(
    item,
    asset="data",
    property_name="raster:bands",
    sub_property_name="classification:classes",
):
    """
    Extract the property from the asset of the item.

    Parameters
    ----------
    item : pystac.Item
        The item to extract the property from.
    asset : str, optional
        The asset name.
    property_name : str, optional
        The property name.
    sub_property_name : str, optional
        The sub property name.

    Returns
    -------
    pandas.DataFrame
        The dataframe containing the property.
    """
    df = pd.DataFrame()
    properties = {}

    if item is not None and item.assets is not None:
        asset = item.assets.get(asset)
        if asset is not None and asset.to_dict() is not None:
            try:
                properties = asset.to_dict()[property_name]
            except NameError:
                print(
                    f'No property "{property_name}" has been found in the asset "{asset}".'
                )
                return None

    property_as_list = {}

    # find the corresponding property in bands
    for property in properties:
        if sub_property_name in property:
            property_as_list = property[sub_property_name]
            break

    # build the dataframe from property list
    for data_dict in property_as_list:
        df = df.append(data_dict, ignore_index=True)

    return df
