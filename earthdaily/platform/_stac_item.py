import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import pystac
from pystac import Item

from earthdaily._api_requester import APIRequester
from earthdaily._http_client import HTTPRequest
from earthdaily.exceptions import EDSAPIError
from earthdaily.platform._stac_item_downloader import ItemDownloader


class ReturnFormat(Enum):
    """Enumeration for different return formats."""

    JSON = "json"
    DICT = "dict"
    PYSTAC = "pystac"


@dataclass
class StacItemService:
    """
    Service for managing STAC items in the Earth Data Store platform.

    This service provides methods to create, read, update, and delete STAC items
    within collections, with configurable return formats.
    """

    def __init__(self, api_requester: APIRequester):
        """
        Initialize the StacItemService.

        Parameters:
        -----------
        api_requester : APIRequester
            An instance of APIRequester used to send HTTP requests to the EDS API.
        """
        self.api_requester = api_requester

    def _validate_return_format(self, return_format: str) -> ReturnFormat:
        """
        Validate and convert return format string to ReturnFormat enum.

        Parameters:
        -----------
        return_format : str
            The return format as string.

        Returns:
        --------
        ReturnFormat
            The validated ReturnFormat enum.

        Raises:
        -------
        ValueError
            If the return format is not supported.
        """
        try:
            return ReturnFormat(return_format.lower())
        except ValueError:
            raise ValueError(f"Unsupported return format: {return_format}. Supported formats: json, dict, pystac")

    def _prepare_item_data(self, item_data: Union[dict[str, Any], Any]) -> dict[str, Any]:
        """
        Convert item data to dictionary format for API requests.

        Parameters:
        -----------
        item_data : Union[dict[str, Any], Any]
            The item data as dict or pystac.Item object.

        Returns:
        --------
        dict[str, Any]
            The item data as dictionary.

        Raises:
        -------
        ValueError
            If the item data format is not supported.
        """
        if isinstance(item_data, dict):
            return item_data

        # Check if it's a pystac Item object
        if hasattr(item_data, "to_dict") and callable(getattr(item_data, "to_dict")):
            try:
                return item_data.to_dict()
            except Exception as e:
                raise ValueError(f"Failed to convert pystac Item to dict: {e}")

        raise ValueError("item_data must be a dictionary or pystac.Item object")

    def _send_request(self, method: str, endpoint: str, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Send a request to a platform endpoint.

        Parameters:
        -----------
        method : str
            HTTP method for the request (e.g., "GET", "POST", "PUT", "DELETE").
        endpoint : str
            The specific endpoint to send the request to.
        data : Optional[dict[str, Any]], optional
            The payload to be sent in the request.

        Returns:
        --------
        dict[str, Any]
            The response from the platform API.

        Raises:
        -------
        EDSAPIError
            If the API returns an error response.
        """
        url = f"{self.api_requester.base_url}/platform/v1/stac/{endpoint}"
        request = HTTPRequest(method=method, url=url, body=data)
        response = self.api_requester.send_request(request)

        if response.status_code not in [200, 201, 204]:
            raise EDSAPIError(f"API request failed with status {response.status_code}: {response.body}")

        return response.body

    def _format_response(
        self, response_data: dict[str, Any], return_format: ReturnFormat
    ) -> Union[str, dict[str, Any], Any]:
        """
        Format the response data according to the specified return format.

        Parameters:
        -----------
        response_data : dict[str, Any]
            The raw response data from the API.
        return_format : ReturnFormat
            The desired format for the return value.

        Returns:
        --------
        Union[str, dict[str, Any], Any]
            The formatted response data.

        Raises:
        -------
        ImportError
            If pystac is not available when PYSTAC format is requested.
        """
        if return_format == ReturnFormat.JSON:
            return json.dumps(response_data, indent=2)
        elif return_format == ReturnFormat.DICT:
            return response_data
        elif return_format == ReturnFormat.PYSTAC:
            try:
                return pystac.Item.from_dict(response_data)
            except ImportError:
                raise ImportError(
                    "pystac library is required for PYSTAC return format. Install with: pip install pystac"
                )
        else:
            raise ValueError(f"Unsupported return format: {return_format}")

    def get_item(
        self, collection_id: str, item_id: str, return_format: str = "dict"
    ) -> Union[str, dict[str, Any], Any]:
        """
        Get a STAC item from a collection.

        Parameters:
        -----------
        collection_id : str
            The ID of the collection containing the item.
        item_id : str
            The ID of the item to retrieve.
        return_format : str, optional
            The format to return the response in. Defaults to "dict".
            Supported formats: "dict", "json", "pystac".

        Returns:
        --------
        Union[str, dict[str, Any], Any]
            The STAC item in the specified format.

        Raises:
        -------
        EDSAPIError
            If the API request fails.
        ValueError
            If the return format is not supported.
        """
        validated_format = self._validate_return_format(return_format)
        endpoint = f"collections/{collection_id}/items/{item_id}"
        response_data = self._send_request("GET", endpoint)
        return self._format_response(response_data, validated_format)

    def create_item(
        self, collection_id: str, item_data: Union[dict[str, Any], Any], return_format: str = "dict"
    ) -> Union[str, dict[str, Any], Any]:
        """
        Create a new STAC item in a collection.

        Parameters:
        -----------
        collection_id : str
            The ID of the collection to create the item in.
        item_data : Union[dict[str, Any], Any]
            The STAC item data to create. Can be a dictionary or pystac.Item object.
        return_format : str, optional
            The format to return the response in. Defaults to "dict".
            Supported formats: "dict", "json", "pystac".

        Returns:
        --------
        Union[str, dict[str, Any], Any]
            The created STAC item in the specified format.

        Raises:
        -------
        EDSAPIError
            If the API request fails.
        ValueError
            If the return format or item data format is not supported.
        """
        validated_format = self._validate_return_format(return_format)
        prepared_data = self._prepare_item_data(item_data)
        endpoint = f"collections/{collection_id}/items"
        response_data = self._send_request("POST", endpoint, prepared_data)
        return self._format_response(response_data, validated_format)

    def update_item(
        self,
        collection_id: str,
        item_id: str,
        item_data: Union[dict[str, Any], Any],
        return_format: str = "dict",
    ) -> Union[str, dict[str, Any], Any]:
        """
        Update an existing STAC item in a collection.

        Parameters:
        -----------
        collection_id : str
            The ID of the collection containing the item.
        item_id : str
            The ID of the item to update.
        item_data : Union[dict[str, Any], Any]
            The updated STAC item data. Can be a dictionary or pystac.Item object.
        return_format : str, optional
            The format to return the response in. Defaults to "dict".
            Supported formats: "dict", "json", "pystac".

        Returns:
        --------
        Union[str, dict[str, Any], Any]
            The updated STAC item in the specified format.

        Raises:
        -------
        EDSAPIError
            If the API request fails.
        ValueError
            If the return format or item data format is not supported.
        """
        validated_format = self._validate_return_format(return_format)
        prepared_data = self._prepare_item_data(item_data)
        endpoint = f"collections/{collection_id}/items/{item_id}"
        response_data = self._send_request("PUT", endpoint, prepared_data)
        return self._format_response(response_data, validated_format)

    def delete_item(self, collection_id: str, item_id: str) -> None:
        """
        Delete a STAC item from a collection.

        Parameters:
        -----------
        collection_id : str
            The ID of the collection containing the item.
        item_id : str
            The ID of the item to delete.

        Returns:
        --------
        None

        Raises:
        -------
        EDSAPIError
            If the API request fails.
        """
        endpoint = f"collections/{collection_id}/items/{item_id}"
        self._send_request("DELETE", endpoint)

    def download_assets(
        self,
        item: Union[dict, Item, str],
        asset_keys: Optional[list] = None,
        output_dir: str = "./downloaded_assets",
        max_workers: int = 3,
        quiet: bool = False,
        continue_on_error: bool = True,
        href_type: str = "href",
    ) -> dict[str, Any]:
        """
        Download assets from a STAC item.

        Parameters:
        -----------
        item : Union[dict[str, Any], Any]
            The STAC item containing the assets to download.
            Can be a dictionary, pystac.Item object, or an item ID.
        asset_keys : Optional[list], optional
            List of asset keys to download. If None, all assets will be downloaded.
        output_dir : str, optional
            Directory to save downloaded files. Defaults to "./downloaded_assets".
        max_workers : int, optional
            Maximum number of concurrent downloads. Defaults to 3.
        quiet : bool, optional
            Whether to suppress progress bars. Defaults to False.
        continue_on_error : bool, optional
            Whether to continue downloading if an error occurs. Defaults to True.
        href_type : str
            Specifies which href to use from asset's alternate URLs.
            Example: "download" would look for item.assets[key].alternate.download.href
            If None, uses the default asset href. Defaults to "href".

        Returns:
        --------
        dict[str, Any]
            Dictionary mapping asset keys to downloaded file paths.

        Raises:
        -------
        ValueError
            If the item cannot be resolved or no assets are found.
        EDSAPIError
            If the API request to get the item fails.
        """
        # Handle the case where item is a string (collection_id/item_id)
        if isinstance(item, str):
            collection_id, item_id = None, None
            parts = item.split("/", 1)
            if len(parts) == 2:
                collection_id, item_id = parts

            if not collection_id or not item_id:
                raise ValueError(
                    "When providing an STAC item as a string, it must be in the format 'collection_id/item_id'"
                )
            item = self.get_item(collection_id=collection_id, item_id=item_id, return_format="dict")

        downloader = ItemDownloader(max_workers=max_workers, api_requester=self.api_requester)

        if not isinstance(item, (Item, dict)):
            raise ValueError("Item must be a PySTAC Item or a dictionary representation of a STAC Item")

        return downloader.download_assets(
            item=item,
            asset_keys=asset_keys,
            output_dir=output_dir,
            quiet=quiet,
            continue_on_error=continue_on_error,
            href_type=href_type,
        )
