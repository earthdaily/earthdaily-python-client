from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO

from earthdaily._api_requester import APIRequester
from earthdaily._eds_config import AssetAccessMode
from earthdaily.platform._bulk_delete import BulkDeleteService
from earthdaily.platform._bulk_insert import BulkInsertService
from earthdaily.platform._bulk_search import BulkSearchService
from earthdaily.platform._stac_item import StacItemService


class PlatformService:
    """
    Represents the Platform Service for interacting with specific platform-related endpoints.

    Attributes:
    -----------
    api_requester : APIRequester
        An instance of APIRequester used to send HTTP requests to the EDS API.
    """

    def __init__(self, api_requester: APIRequester, asset_access_mode: AssetAccessMode):
        """
        Initialize the PlatformService.

        Parameters:
        -----------
        api_requester : APIRequester
            An instance of APIRequester used to send HTTP requests to the EDS API.
        asset_access_mode : AssetAccessMode
            The mode of access for assets. Defaults to AssetAccessMode.PRESIGNED_URLS.
        """
        self.api_requester = api_requester
        self.bulk_search = BulkSearchService(api_requester)
        self.bulk_insert = BulkInsertService(api_requester)
        self.bulk_delete = BulkDeleteService(api_requester)
        self.stac_item = StacItemService(api_requester)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **api_requester.headers,
        }
        if asset_access_mode == AssetAccessMode.PROXY_URLS:
            headers["X-Proxy-Asset-Urls"] = "True"
        elif asset_access_mode == AssetAccessMode.PRESIGNED_URLS:
            headers["X-Signed-Asset-Urls"] = "True"
        if api_requester.auth:
            headers["Authorization"] = f"Bearer {api_requester.auth.get_token()}"
        self.pystac_client = Client.open(
            f"{api_requester.base_url}/platform/v1/stac",
            stac_io=StacApiIO(max_retries=3),
            headers=headers,
        )
