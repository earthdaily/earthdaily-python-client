import re
from abc import ABC, abstractmethod
from typing import Optional, Pattern
from urllib.parse import urlparse

from earthdaily._api_requester import APIRequester
from earthdaily._eds_logging import LoggerConfig

logger = LoggerConfig(logger_name=__name__).get_logger()


class AssetResolver(ABC):
    """Base class for asset resolvers that handle specific domains or protocols."""

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Check if this resolver can handle the given URL."""
        pass

    @abstractmethod
    def get_download_url(self, url: str) -> str:
        """
        Transform the asset URL if needed before download.

        Returns:
            str: The URL to use for downloading
        """
        pass

    @abstractmethod
    def get_headers(self, url: str) -> dict[str, str]:
        """
        Get headers needed for downloading from this service.

        Returns:
            dict[str, str]: Headers to include in the download request
        """
        pass


class DefaultResolver(AssetResolver):
    """Default resolver that handles standard HTTP/HTTPS URLs with no special requirements."""

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in ["http", "https"]

    def get_download_url(self, url: str) -> str:
        return url

    def get_headers(self, url: str) -> dict[str, str]:
        return {}


class DomainPatternResolver(AssetResolver):
    """Base class for resolvers that use domain pattern matching."""

    domain_pattern: Optional[Pattern] = None

    def can_handle(self, url: str) -> bool:
        if not self.domain_pattern:
            return False

        parsed = urlparse(url)
        return bool(self.domain_pattern.match(parsed.netloc))


class EarthDailyAPIResolver(DomainPatternResolver):
    """Resolver for EarthDaily API domains."""

    domain_pattern = re.compile(r".*\.earthdaily\.com$")

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def get_download_url(self, url: str) -> str:
        return url

    def get_headers(self, url: str) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class ResolverRegistry:
    """Registry for asset resolvers."""

    def __init__(self):
        self._resolvers: list[AssetResolver] = []

    def register(self, resolver: AssetResolver) -> None:
        """Register a resolver."""
        self._resolvers.append(resolver)

    def get_resolver(self, url: str) -> AssetResolver:
        """Get the appropriate resolver for a URL."""
        for resolver in self._resolvers:
            if resolver.can_handle(url):
                return resolver
        return DefaultResolver()


resolver_registry = ResolverRegistry()

# The order of registration matters; more specific resolvers should be registered before more general ones.
resolver_registry.register(EarthDailyAPIResolver())
resolver_registry.register(DefaultResolver())


def get_resolver_for_url(url: str, api_requester: Optional[APIRequester] = None) -> AssetResolver:
    """Get the appropriate resolver for a URL."""
    resolver = resolver_registry.get_resolver(url)
    if isinstance(resolver, EarthDailyAPIResolver) and api_requester:
        resolver.api_key = api_requester.auth.get_token() if api_requester.auth else None
    return resolver
