from importlib.metadata import PackageNotFoundError, version

# Enable namespace package support for dual installation
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

try:
    __version__ = version("earthdaily")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Authentication-related imports
from earthdaily._api_requester import APIRequester
from earthdaily._auth_client import Authentication

# Client and config-related imports
from earthdaily._eds_client import EDSClient
from earthdaily._eds_config import EDSConfig

# HTTP-related imports
from earthdaily._http_client import HTTPClient, HTTPRequest, HTTPResponse

__all__ = [
    "Authentication",
    "APIRequester",
    "HTTPClient",
    "HTTPRequest",
    "HTTPResponse",
    "EDSClient",
    "EDSConfig",
]
