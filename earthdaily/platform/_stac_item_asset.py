from typing import Callable
from urllib.parse import urlparse

from earthdaily._api_requester import APIRequester
from earthdaily.platform._download_resolvers import EarthDataResolver, EUMETSATResolver, S3Resolver
from earthdaily.platform._resolver_base import (
    AssetResolver,
    DefaultResolver,
    DomainPatternResolver,
    EarthDailyAPIResolver,
)

__all__ = [
    "AssetResolver",
    "DefaultResolver",
    "DomainPatternResolver",
    "EarthDailyAPIResolver",
    "ResolverNotConfiguredError",
    "ResolverRegistry",
    "get_resolver_for_url",
    "resolver_registry",
]

RESOLVER_CLASSES: tuple[type[AssetResolver], ...] = (
    EarthDailyAPIResolver,
    EarthDataResolver,
    EUMETSATResolver,
    S3Resolver,
)


class ResolverRegistry:
    """Holds explicitly configured resolver instances.

    Supports a one-time *lazy configure* callback so that heavy
    initialisation (e.g. credential fetching) only runs when the
    first download is actually attempted.
    """

    def __init__(self) -> None:
        self._resolvers: list[AssetResolver] = []
        self._lazy_configure: Callable[[], None] | None = None

    def set_lazy_configure(self, setup: Callable[[], None]) -> None:
        """Register a one-time callback that runs before the first resolver lookup."""
        self._lazy_configure = setup

    def _run_lazy_configure(self) -> None:
        if self._lazy_configure is not None:
            setup = self._lazy_configure
            self._lazy_configure = None
            setup()

    def register(self, resolver: AssetResolver) -> None:
        """Add a configured resolver instance, replacing any existing one of the same type."""
        self._resolvers = [r for r in self._resolvers if type(r) is not type(resolver)]
        self._resolvers.append(resolver)

    def get_resolver(self, url: str) -> AssetResolver | None:
        """Return the first configured resolver that can handle *url*, or None.

        Runs the lazy configure callback (if any) on the first call.
        """
        self._run_lazy_configure()
        for resolver in self._resolvers:
            if resolver.can_handle(url):
                return resolver
        return None


class ResolverNotConfiguredError(Exception):
    """Raised when a URL matches a resolver that requires credentials."""

    def __init__(self, resolver_cls: type, url: str) -> None:
        name = resolver_cls.__name__
        super().__init__(
            f"URL '{url}' requires {name} but no credentials have been configured. "
            f"Provide the required credentials via EDSConfig or environment variables."
        )


resolver_registry = ResolverRegistry()


def _class_can_handle(cls: type[AssetResolver], url: str) -> bool:
    """Check whether a resolver *class* can handle *url* (without a full instance)."""
    pattern = getattr(cls, "domain_pattern", None)
    if pattern is not None:
        parsed = urlparse(url)
        return bool(pattern.match(parsed.netloc))
    if cls.can_handle is not AssetResolver.can_handle:
        try:
            probe = object.__new__(cls)
            return probe.can_handle(url)
        except Exception:
            return False
    return False


def get_resolver_for_url(url: str, api_requester: APIRequester | None = None) -> AssetResolver:
    """Find the right resolver for *url*.

    Resolution order:
    1. Configured instances in the registry.
    2. URL-based auto-discovery across known resolver classes.  Classes with
       ``needs_credentials = True`` raise ``ResolverNotConfiguredError``;
       others are instantiated on the fly and cached.
    3. ``DefaultResolver`` as the final HTTP(S) fallback.
    """
    resolver = resolver_registry.get_resolver(url)
    if resolver is not None:
        if isinstance(resolver, EarthDailyAPIResolver) and api_requester:
            resolver.api_key = api_requester.auth.get_token() if api_requester.auth else None
        return resolver

    for cls in RESOLVER_CLASSES:
        if _class_can_handle(cls, url):
            if getattr(cls, "needs_credentials", False):
                config = api_requester.config if api_requester else None
                from_config = getattr(cls, "from_config", None)
                if from_config and config:
                    instance = from_config(config)
                    if instance is not None:
                        resolver_registry.register(instance)
                        return instance
                raise ResolverNotConfiguredError(cls, url)
            instance = cls()
            if isinstance(instance, EarthDailyAPIResolver) and api_requester:
                instance.api_key = api_requester.auth.get_token() if api_requester.auth else None
            resolver_registry.register(instance)
            return instance

    return DefaultResolver()
