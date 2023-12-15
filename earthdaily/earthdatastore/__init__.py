import json
import logging
import operator
import os
import warnings
import time
import geopandas as gpd
import pandas as pd
import psutil
import requests
import xarray as xr
from pystac.item_collection import ItemCollection
from pystac_client import Client
from odc import stac
from . import _scales_collections, cube_utils, mask
from .cube_utils import datacube, metacube, _datacubes

__all__ = ["datacube", "metacube", "xr", "stac"]

logging.getLogger("earthdaily-earthdatastore")


def post_query_items(items, query):
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


def _get_client(config=None, presign_urls=True):
    if config is None:
        config = os.getenv
    else:
        if isinstance(config, str):
            config = json.load(open(config, "rb"))
        config = config.get
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
    tokens = json.loads(token_response.text)
    headers = {"Authorization": f"bearer {tokens['access_token']}"}
    if presign_urls:
        headers["X-Signed-Asset-Urls"] = "True"

    client = Client.open(
        eds_url,
        headers=headers,
    )
    return client


class StacCollectionExplorer:
    def __init__(self, client, collection):
        self.client = client
        self.collection = collection
        self.client_collection = self.client.get_collection(self.collection)
        self.item = self.__first_item()
        self.properties = self.client_collection.to_dict()

    def __first_item(self):
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

    def _update_search_kwargs_for_ag_cloud_mask(self, search_kwargs, collections):
        search_kwargs = search_kwargs.copy()
        # to get only items that have a ag_cloud_mask
        ag_query = {"eda:ag_cloud_mask_available": {"eq": True}}
        target_param = "query"

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
        intersects: (gpd.GeoDataFrame, str, dict) = None,
        bbox=None,
        mask_with: (None, str) = None,
        mask_statistics: bool | int = False,
        clear_cover: (int, float) = None,
        prefer_alternate: (str, False) = "download",
        search_kwargs: dict = {},
        add_default_scale_factor: bool = True,
        common_band_names=True,
        cross_calibration_collection: (None | str) = None,
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
        mask_with : (None, str), optional
            "native" mask, or "ag_cloud_mask".
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
        if isinstance(collections, str):
            collections = [collections]

        if mask_with:
            if mask_with not in mask._available_masks:
                raise ValueError(
                    f"Specified mask '{mask_with}' is not available. Currently available masks provider are : {mask._available_masks}"
                )
            elif assets is not None:
                mask_with = mask._native_mask_def_mapping.get(collections[0], None)
                assets.append(mask_with)
            if mask_with == "ag_cloud_mask":
                search_kwargs = self._update_search_kwargs_for_ag_cloud_mask(
                    search_kwargs, collections[0]
                )

        if intersects is not None:
            intersects = cube_utils.GeometryManager(intersects).to_geopandas()
        items = self.search(
            collections=collections,
            bbox=bbox,
            intersects=intersects,
            datetime=datetime,
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

        xr_datacube = datacube(
            items,
            intersects=intersects,
            bbox=bbox,
            assets=assets,
            common_band_names=common_band_names,
            cross_calibration_items=xcal_items,
            **kwargs,
        )
        if mask_with:
            if clear_cover and mask_statistics is False:
                mask_statistics = True
            mask_kwargs = dict(mask_statistics=mask_statistics)
            if mask_with == "ag_cloud_mask":
                acm_items = self.ag_cloud_mask_items(items)
                acm_datacube = datacube(
                    acm_items,
                    intersects=intersects,
                    bbox=bbox,
                    groupby_date="max",
                    geobox=xr_datacube.odc.geobox
                    if hasattr(xr_datacube, "odc")
                    else None,
                )
                xr_datacube["time"] = xr_datacube.time.astype("M8[s]")
                acm_datacube["time"] = acm_datacube.time.astype("M8[s]")
                acm_datacube = cube_utils._match_xy_dims(acm_datacube, xr_datacube)
                mask_kwargs.update(acm_datacube=acm_datacube)
            else:
                mask_assets = mask._native_mask_asset_mapping[collections[0]]
                if "groupby_date" in kwargs:
                    kwargs["groupby_date"] = "max"
                if "resolution" in kwargs:
                    kwargs.pop("resolution")
                if "epsg" in kwargs:
                    kwargs.pop("epsg")
                if "geobox" in kwargs:
                    kwargs.pop("geobox")

                clouds_datacube = datacube(
                    items,
                    intersects=intersects,
                    bbox=bbox,
                    assets=[mask_assets],
                    resampling=0,
                    geobox=xr_datacube.odc.geobox
                    if hasattr(xr_datacube, "odc")
                    else None,
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

    def search(
        self,
        collections: str | list,
        intersects: gpd.GeoDataFrame = None,
        bbox=None,
        post_query=None,
        prefer_alternate=None,
        add_default_scale_factor=False,
        assets=None,
        **kwargs,
    ):
        """
        search using pystac client search. Add some features to enhance experience.

        Parameters
        ----------
        collections : str | list
            DESCRIPTION.
        intersects : gpd.GeoDataFrame, optional
            DESCRIPTION. The default is None.
        bbox : TYPE, optional
            DESCRIPTION. The default is None.
        post_query : TYPE, optional
            DESCRIPTION. The default is None.
        prefer_alternate : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        items_collection : TYPE
            DESCRIPTION.

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
        if len(items_collection) == 0:
            raise Warning("No item has been found.")
        return items_collection

    def ag_cloud_mask_items(self, items_collection):
        def ag_cloud_mask_from_items(items):
            products = {}
            for item in items:
                if not item.properties.get("eda:ag_cloud_mask_available"):
                    continue
                collection = item.properties["eda:ag_cloud_mask_collection_id"]
                if products.get(collection, None) is None:
                    products[collection] = []
                products[collection].append(
                    item.properties.get("eda:ag_cloud_mask_item_id")
                )
            return products

        items_id = ag_cloud_mask_from_items(items_collection)
        if len(items_id) == 0:
            raise ValueError("Sorry, no ag_cloud_mask available.")
        collections = list(items_id.keys())
        ids = [x for n in (items_id.values()) for x in n]
        return self.search(collections=collections, ids=ids)


def item_property_to_df(
    item,
    asset="data",
    property_name="raster:bands",
    sub_property_name="classification:classes",
):
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
