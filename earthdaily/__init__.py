from importlib.metadata import PackageNotFoundError, version

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

# Platform-related imports
from earthdaily.platform import PlatformService

__all__ = [
    "Authentication",
    "APIRequester",
    "HTTPClient",
    "HTTPRequest",
    "HTTPResponse",
    "EDSClient",
    "EDSConfig",
    "PlatformService",
]
