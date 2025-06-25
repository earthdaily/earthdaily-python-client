import json
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from earthdaily._auth_client import Authentication
from earthdaily._eds_config import AssetAccessMode, EDSConfig
from earthdaily._http_client import HTTPClient, HTTPRequest, HTTPResponse


class APIRequester:
    """
    Handles high-level API requests, managing authentication and service interactions.

    This class is responsible for preparing API requests and delegating the actual HTTP
    calls to the HTTPClient. It also handles adding authorization headers.

    Attributes:
    ----------
    config: EDSConfig
        The configuration object containing settings for the API requester.
    base_url: str
        The base URL for all API calls.
    auth: Authentication
        The authentication manager to obtain tokens for API requests.
    http_client: HTTPClient
        The HTTP client responsible for making actual HTTP requests.
    headers: dict
        The default headers to be included in every request.

    Methods:
    -------
    send_request(request: HTTPRequest) -> HTTPResponse:
        Sends an HTTP request using the HTTPClient and returns the response. It automatically
        adds an authorization header to the request using a valid token from the Authentication instance.
    """

    def __init__(self, config: EDSConfig, auth: Optional[Authentication] = None):
        """
        Initializes a new instance of APIRequester.

        Parameters:
        ----------
        config: EDSConfig
            The configuration object containing settings for the API requester.
        auth: Authentication
            The authentication manager to obtain tokens for API requests.
        """
        self.config = config
        self.base_url = config.base_url
        self.auth = auth
        self.http_client = HTTPClient()
        self.headers = self._generate_headers()

    def _generate_headers(self) -> dict[str, str]:
        """
        Generates HTTP headers for the EarthDaily Python client.

        This method collects metadata about the client environment, including the client version,
        Python version, and system information, to construct headers that are used in HTTP requests.

        Returns:
            dict[str, str]: A dictionary containing the following headers:
                - "X-EDA-Client-User-Agent": A JSON string with detailed client metadata.
                - "User-Agent": A user agent string with the client version and system information.

        Metadata collected:
            - client_version: The version of the EarthDaily Python client.
            - language: The programming language used (Python).
            - publisher: The publisher of the client (EarthDaily).
            - http_library: The HTTP library used (requests).
            - python_version: The version of Python being used.
            - platform: The platform information.
            - system_info: Detailed system information.

        If any metadata collection fails, default values are used to ensure the headers are always generated.
        """

        try:
            client_version = version("earthdaily")
        except PackageNotFoundError:
            client_version = "0.0.0"

        try:
            python_version = platform.python_version()
            system_platform = platform.platform()
            uname_info = " ".join(platform.uname())
        except Exception:
            python_version = "(unknown)"
            system_platform = "(unknown)"
            uname_info = "(unknown)"

        user_agent = f"EarthDaily-Python-Client/{client_version} (Python/{python_version}; {system_platform})"

        client_metadata = {
            "client_version": client_version,
            "language": "Python",
            "publisher": "EarthDaily",
            "http_library": "requests",
            "python_version": python_version,
            "platform": system_platform,
            "system_info": uname_info,
        }

        headers = {
            "X-EDA-Client-User-Agent": json.dumps(client_metadata),
            "User-Agent": user_agent,
        }

        if self.config.asset_access_mode == AssetAccessMode.PRESIGNED_URLS:
            headers["X-Signed-Asset-Urls"] = "True"
        elif self.config.asset_access_mode == AssetAccessMode.PROXY_URLS:
            headers["X-Proxy-Asset-Urls"] = "True"

        return headers

    def send_request(self, request: HTTPRequest) -> HTTPResponse:
        """
        Sends an HTTP request using the HTTPClient, adding authorization if necessary.

        This method updates the 'Authorization' header of the request object with a 'Bearer' token
        obtained from the Authentication instance. It then sends the modified request through the
        HTTP client and returns the response.

        Parameters:
        ----------
        request: HTTPRequest
            The HTTPRequest object containing method, URL, headers, and body.

        Returns:
        -------
        HTTPResponse:
            The HTTPResponse object containing status code, body, and headers.
        """

        request.headers.update(self.headers)
        if self.auth:
            request.headers["Authorization"] = f"Bearer {self.auth.get_token()}"
        response = self.http_client.send(request)
        return response
