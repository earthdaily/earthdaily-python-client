from earthdatastore._auth_client import Authentication
from earthdatastore._http_client import HTTPClient, HTTPRequest, HTTPResponse


class APIRequester:
    """
    Handles high-level API requests, managing authentication and service interactions.

    This class is responsible for preparing API requests and delegating the actual HTTP
    calls to the HTTPClient. It also handles adding authorization headers.

    Attributes:
    ----------
    base_url: str
        The base URL for all API calls.
    auth: Authentication
        The authentication manager to obtain tokens for API requests.
    http_client: HTTPClient
        The HTTP client responsible for making actual HTTP requests.

    Methods:
    -------
    send_request(request: HTTPRequest) -> HTTPResponse:
        Sends an HTTP request using the HTTPClient and returns the response. It automatically
        adds an authorization header to the request using a valid token from the Authentication instance.
    """

    def __init__(self, base_url: str, auth: Authentication):
        """
        Initializes a new instance of APIRequester.

        Parameters:
        ----------
        base_url: str
            The base URL for all API calls.
        auth: Authentication
            The authentication manager to obtain tokens for API requests.
        """
        self.base_url = base_url
        self.auth = auth
        self.http_client = HTTPClient()

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

        request.headers["Authorization"] = f"Bearer {self.auth.get_token()}"
        response = self.http_client.send(request)
        return response
