from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO

from earthdaily._api_requester import APIRequester


class PlatformService:
    """
    Represents the Platform Service for interacting with specific platform-related endpoints.

    Attributes:
    -----------
    api_requester : APIRequester
        An instance of APIRequester used to send HTTP requests to the EDS API.
    """

    def __init__(self, api_requester: APIRequester, pre_sign_urls: bool):
        """
        Initialize the PlatformService.

        Parameters:
        -----------
        api_requester : APIRequester
            An instance of APIRequester used to send HTTP requests to the EDS API.
        pre_sign_urls : bool, optional
            A flag indicating whether to use pre-signed URLs for asset access.
        """
        self.api_requester = api_requester

        self.pystac_client = Client.open(
            f"{api_requester.base_url}/platform/v1/stac",
            stac_io=StacApiIO(max_retries=3),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **api_requester.headers,
                "Authorization": f"Bearer {api_requester.auth.get_token()}",
                "X-Signed-Asset-Urls": str(pre_sign_urls),
            },
        )
