from typing import Any, Dict, Optional

import requests
from requests.structures import CaseInsensitiveDict


class HTTPRequest:
    """
    Represents an HTTP request with method, URL, headers, and body.

    Attributes:
    ----------
    method: str
        The HTTP method (e.g., "GET", "POST", "PUT", etc.).
    url: str
        The full URL or endpoint to send the request to.
    headers: dict
        The headers to be included in the request.
    body: dict
        The payload of the request (for POST, PUT requests, etc.).
    """

    def __init__(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
    ):
        self.method = method
        self.url = url
        self.headers = headers or {}
        if headers is None or "Content-Type" not in headers:
            self.headers["Content-Type"] = "application/json"
        self.body = body or {}
        self.timeout_seconds: int = 15


class HTTPResponse:
    """
    Represents an HTTP response with status code, body, and headers.

    Attributes:
    ----------
    status_code: int
        The HTTP status code of the response.
    body: dict
        The body of the response (assumed to be JSON).
    headers: dict
        The headers returned by the server.
    """

    def __init__(self, status_code: int, body: Dict[str, Any], headers: CaseInsensitiveDict[str]):
        self.status_code = status_code
        self.body = body
        self.headers = headers


class HTTPClient:
    """
    Responsible for making actual HTTP requests using the `requests` library.

    This class abstracts the HTTP communication and interacts directly with external
    services using the given `HTTPRequest` object.

    Methods:
    -------
    send(request: HTTPRequest) -> HTTPResponse:
        Sends the HTTP request and returns the HTTP response.
    """

    def send(self, request: HTTPRequest) -> HTTPResponse:
        """
        Sends an HTTP request and returns the HTTP response.

        Parameters:
        ----------
        request: HTTPRequest
            The HTTPRequest object containing method, URL, headers, and body.

        Returns:
        -------
        HTTPResponse:
            The response object containing status code, body, and headers.
        """
        response = requests.request(
            method=request.method,
            url=request.url,
            headers=request.headers,
            json=request.body,
            timeout=request.timeout_seconds,
        )

        # Handle responses with no content (204) or empty bodies
        try:
            body = response.json() if response.content else {}
        except (ValueError, requests.exceptions.JSONDecodeError):
            body = {}

        return HTTPResponse(
            status_code=response.status_code,
            body=body,
            headers=response.headers,
        )
