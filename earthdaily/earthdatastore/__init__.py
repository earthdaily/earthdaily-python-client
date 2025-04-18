# mypy: ignore-errors
# TODO (v1): Fix type issues and remove 'mypy: ignore-errors' after verifying non-breaking changes
import json
import logging
import operator
import os
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import requests
import toml
import xarray as xr
from odc import stac
from pystac.item_collection import ItemCollection
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

from . import _scales_collections, cube_utils, mask
from .cube_utils import _datacubes, asset_mapper, datacube, metacube
from .parallel_search import NoItemsFoundError, parallel_search

__all__ = ["datacube", "metacube", "xr", "stac"]

logging.getLogger("earthdaily-earthdatastore")


@dataclass
class EarthDataStoreConfig:
    auth_url: Optional[str] = None
    client_secret: Optional[str] = None
    client_id: Optional[str] = None
    eds_url: str = "https://api.earthdaily.com/platform/v1/stac"
    access_token: Optional[str] = None


def apply_single_condition(
    item_value, condition_op: str, condition_value: Any | list[Any]
) -> bool:
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


def validate_property_condition(
    item: Any, property_name: str, conditions: dict[str, Any]
) -> bool:
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
        apply_single_condition(
            item.properties.get(property_name), condition_op, condition_value
        )
        for condition_op, condition_value in conditions.items()
    )


def filter_items(items: list[Any], query: dict[str, dict[str, Any]]) -> list[Any]:
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
            validate_property_condition(item, property_name, conditions)
            for property_name, conditions in query.items()
        )
    ]


def post_query_items(
    items: ItemCollection | list[Any], query: dict[str, dict[str, Any]]
) -> ItemCollection:
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
    return ItemCollection(
        filtered_items
    )  # Assuming ItemCollection is imported/defined elsewhere


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
                            item.collection_id if item.collection_id else "", [{}]
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
        config: str | dict = None,
        presign_urls=True,
        asset_proxy_enabled=False,
        client_version: str = "0.0.0",
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
        client_version : str, optional
            The version of the client.
            Uses the current version by default.

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

        self._client_version = client_version
        self._client = None
        self.__auth_config = config
        self.__presign_urls = presign_urls
        self.__asset_proxy_enabled = asset_proxy_enabled
        self._first_items_: dict = {}
        self._staccollectionexplorer = {}
        self.__time_eds_log = time.time()
        self._client = self.client

    @classmethod
    def from_credentials(
        cls,
        json_path: Optional[Path] = None,
        toml_path: Optional[Path] = None,
        profile: Optional[str] = None,
        client_version: str = "0.0.0",
        presign_urls: bool = True,
        asset_proxy_enabled: bool = False,
    ) -> "Auth":
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
        client_version : str, optional
            The version of the client.
            Uses the current version by default.
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
            client_version=client_version,
        )

    @classmethod
    def read_credentials(
        cls,
        json_path: Optional[Path] = None,
        toml_path: Optional[Path] = None,
        profile: Optional[str] = None,
    ) -> dict:
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
        try:
            if json_path is not None:
                config = cls.read_credentials_from_json(json_path=json_path)

            elif toml_path is not None:
                config = cls.read_credentials_from_toml(
                    toml_path=toml_path, profile=profile
                )

            elif (
                os.getenv("EDS_AUTH_URL")
                and os.getenv("EDS_SECRET")
                and os.getenv("EDS_CLIENT_ID")
            ):
                config = cls.read_credentials_from_environment()

            else:
                config = cls.read_credentials_from_ini(profile=profile)
        except Exception:
            raise NotImplementedError("Credentials weren't found.")
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

        if profile is None:
            profile = "default"
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
    def read_credentials_from_toml(
        cls, toml_path: Optional[Path] = None, profile: Optional[str] = None
    ) -> dict:
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
        if toml_path is None or not toml_path.exists():
            raise FileNotFoundError(
                f"Credentials file {toml_path} not found. Make sure the path is valid"
            )

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
        if not config.auth_url or not config.client_id or not config.client_secret:
            raise ValueError(
                "Authentication credentials (auth_url, client_id, client_secret) must not be None"
            )
        response = requests.post(
            config.auth_url,
            data={"grant_type": "client_credentials"},
            allow_redirects=False,
            auth=(config.client_id, config.client_secret),
        )
        response.raise_for_status()
        return response.json()["access_token"]

    def _config_parser(self, config=None) -> EarthDataStoreConfig:
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
            eds_url = config(
                "EDS_API_URL", "https://api.earthdaily.com/platform/v1/stac"
            )
            if auth_url is None or client_secret is None or client_id is None:
                raise AttributeError(
                    "You need to have env : EDS_AUTH_URL, EDS_SECRET and EDS_CLIENT_ID"
                )
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

        try:
            python_version = platform.python_version()
            system_platform = platform.platform()
            uname_info = " ".join(platform.uname())
        except Exception:
            python_version = "(unknown)"
            system_platform = "(unknown)"
            uname_info = "(unknown)"

        user_agent = f"EarthDaily-Python-Client/{self._client_version} (Python/{python_version}; {system_platform})"

        client_metadata = {
            "client_version": self._client_version,
            "language": "Python",
            "publisher": "EarthDaily",
            "http_library": "requests",
            "python_version": python_version,
            "platform": system_platform,
            "system_info": uname_info,
        }

        headers["User-Agent"] = user_agent
        headers["X-EDA-Client-User-Agent"] = json.dumps(client_metadata)

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
            self._client = self._get_client(
                self.__auth_config, self.__presign_urls, self.__asset_proxy_enabled
            )
            self.__time_eds_log = time.time()

        return self._client

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
        drop_duplicates: str = "first",
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
            raise NotImplementedError(
                "You must set `groupby_date=None` to have properties per item."
            )

        # convert collections to list
        collections = [collections] if isinstance(collections, str) else collections

        # if intersects a geometry, create a GeoDataFrame
        if intersects is not None:
            intersects = cube_utils.GeometryManager(intersects).to_geopandas()
            self.intersects = intersects

        # if mask_with, need to add assets or to get mask item id
        if mask_with:
            sensor_mask = mask_with
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
                sensor_mask = mask_with

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
            drop_duplicates=drop_duplicates,
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
                    drop_duplicates=False,
                )
            except Warning:
                raise Warning(
                    "No cross calibration coefficient available for the specified collections."
                )

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
            kwargs["rescale"] = True
            kwargs["resampling"] = 0

            if clear_cover and mask_statistics is False:
                mask_statistics = True
            mask_kwargs = dict(mask_statistics=False)
            if mask_with == "ag_cloud_mask" or mask_with == "cloud_mask":
                mask_asset_mapping: dict[str, dict[str, str]] = {
                    "ag_cloud_mask": {"agriculture-cloud-mask": "ag_cloud_mask"},
                    "cloud_mask": {"cloud-mask": "cloud_mask"},
                }
                items_mask = self.find_cloud_mask_items(
                    items, cloudmask=mask_with, **cloud_search_kwargs
                )
                ds_mask = datacube(
                    items_mask,
                    intersects=intersects,
                    groupby_date=None,
                    assets=mask_asset_mapping[mask_with],
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
                    mask._native_mask_asset_mapping[
                        collections[0]
                    ]: mask._native_mask_asset_mapping[collections[0]]
                }
                # force resampling at nearest as qualitative value.
                ds_mask = datacube(
                    items,
                    intersects=intersects,
                    groupby_date=None,
                    bbox=list(cube_utils.GeometryManager(intersects).to_bbox()),
                    assets=assets_mask,
                    **kwargs,
                )
                ds_mask = cube_utils._match_xy_dims(ds_mask, ds)
                if intersects is not None:
                    ds_mask = ds_mask.ed.clip(intersects)
                asset_mask_str = mask._native_mask_asset_mapping[collections[0]]
                if asset_mask_str in ds.data_vars:
                    ds = ds.drop_vars(asset_mask_str)
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
            xy = ds[sensor_mask].isel(time=0).size

            null_pixels = ds[sensor_mask].isnull().sum(dim=("x", "y"))
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
            if clear_cover:
                ds = mask.filter_clear_cover(ds, clear_cover)
            ds = ds.drop_vars(sensor_mask)

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
        for idx, asset in enumerate(assets):
            assets[idx] = f"'{asset}'" if "." in asset else asset

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
        drop_duplicates: str = False,
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
        drop_duplicates : bool, str, Optional
            Drop duplicates. Available : "first" or "last". The default is False.
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
            assets = list(
                asset_mapper.AssetMapper()
                .map_collection_assets(collections[0], assets)
                .keys()
            )
            kwargs["fields"] = self._update_search_for_assets(assets)

        if bbox is None and intersects is not None:
            intersects = cube_utils.GeometryManager(intersects).to_intersects(
                crs="4326"
            )
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
        if drop_duplicates:
            from ._filter_duplicate import filter_duplicate_items

            items_collection = filter_duplicate_items(
                items_collection, keep=drop_duplicates
            )
        return items_collection

    def find_cloud_mask_items(
        self, items_collection, cloudmask="ag_cloud_mask", **kwargs
    ):
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
        kwargs.setdefault("prefer_alternate", "download")

        for items_start_idx in range(0, len(ids_), step):
            items = self.search(
                collections=collections,
                ids=ids_[items_start_idx : items_start_idx + step],
                limit=step,
                drop_duplicates=False,
                **kwargs,
            )
            items_list.extend(list(items))
        return ItemCollection(items_list)
