import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from earthdaily.platform._resolver_base import (
    AssetResolver,
    DefaultResolver,
    EarthDailyAPIResolver,
)
from earthdaily.platform._stac_item_asset import (
    RESOLVER_CLASSES,
    ResolverNotConfiguredError,
    ResolverRegistry,
    _class_can_handle,
    get_resolver_for_url,
    resolver_registry,
)


class TestResolverRegistryNew(unittest.TestCase):
    def test_register_and_get(self):
        registry = ResolverRegistry()
        r = DefaultResolver()
        registry.register(r)
        assert registry.get_resolver("https://example.com/f.tif") is r

    def test_get_returns_none_for_unhandled(self):
        registry = ResolverRegistry()
        assert registry.get_resolver("https://example.com/f.tif") is None

    def test_register_replaces_same_type(self):
        registry = ResolverRegistry()
        r1 = EarthDailyAPIResolver(api_key="old")
        r2 = EarthDailyAPIResolver(api_key="new")
        registry.register(r1)
        registry.register(r2)
        assert len(registry._resolvers) == 1
        assert registry._resolvers[0].api_key == "new"

    def test_register_keeps_different_types(self):
        registry = ResolverRegistry()
        registry.register(EarthDailyAPIResolver())
        registry.register(DefaultResolver())
        assert len(registry._resolvers) == 2

    def test_last_registered_same_type_wins(self):
        registry = ResolverRegistry()
        r1 = DefaultResolver()
        registry.register(r1)
        r2 = DefaultResolver()
        registry.register(r2)
        assert registry.get_resolver("https://example.com/f") is r2


class TestClassCanHandleNew(unittest.TestCase):
    def test_domain_pattern_class(self):
        assert _class_can_handle(EarthDailyAPIResolver, "https://api.earthdaily.com/f")

    def test_domain_pattern_no_match(self):
        assert not _class_can_handle(EarthDailyAPIResolver, "https://example.com/f")

    def test_custom_can_handle_class(self):
        from earthdaily.platform._download_resolvers import S3Resolver

        assert _class_can_handle(S3Resolver, "s3://bucket/key/file.tif")

    def test_custom_can_handle_no_match(self):
        from earthdaily.platform._download_resolvers import S3Resolver

        assert not _class_can_handle(S3Resolver, "https://example.com/file.tif")

    def test_base_abc_returns_false(self):
        assert not _class_can_handle(AssetResolver, "https://example.com/f")

    def test_class_can_handle_exception_returns_false(self):
        class BrokenResolver(AssetResolver):
            def can_handle(self, url):
                raise RuntimeError("boom")

            def get_download_url(self, url):
                return url

            def get_headers(self, url):
                return {}

        assert not _class_can_handle(BrokenResolver, "https://example.com/f")


class TestResolverNotConfiguredErrorNew(unittest.TestCase):
    def test_message_contains_class_name_and_url(self):
        err = ResolverNotConfiguredError(EarthDailyAPIResolver, "https://foo.com/bar")
        assert "EarthDailyAPIResolver" in str(err)
        assert "https://foo.com/bar" in str(err)


class TestLazyConfigure(unittest.TestCase):
    def setUp(self):
        self._original_resolvers = list(resolver_registry._resolvers)
        self._original_lazy_configure = resolver_registry._lazy_configure

    def tearDown(self):
        resolver_registry._resolvers = self._original_resolvers
        resolver_registry._lazy_configure = self._original_lazy_configure

    def test_lazy_configure_runs_on_first_get_resolver(self):
        called = []
        resolver_registry.set_lazy_configure(lambda: called.append(True))
        resolver_registry._resolvers = []

        resolver_registry.get_resolver("https://example.com/f.tif")
        assert called == [True]

    def test_lazy_configure_runs_only_once(self):
        call_count = []
        resolver_registry.set_lazy_configure(lambda: call_count.append(1))
        resolver_registry._resolvers = []

        resolver_registry.get_resolver("https://example.com/a.tif")
        resolver_registry.get_resolver("https://example.com/b.tif")
        assert len(call_count) == 1

    def test_lazy_configure_none_by_default(self):
        registry = ResolverRegistry()
        assert registry._lazy_configure is None
        registry.get_resolver("https://example.com/f.tif")


class TestGetResolverForUrlNew(unittest.TestCase):
    def setUp(self):
        self._original_resolvers = list(resolver_registry._resolvers)
        self._original_lazy_configure = resolver_registry._lazy_configure

    def tearDown(self):
        resolver_registry._resolvers = self._original_resolvers
        resolver_registry._lazy_configure = self._original_lazy_configure

    def test_returns_default_for_plain_http(self):
        resolver_registry._resolvers = []
        r = get_resolver_for_url("https://some-unknown-domain.com/file.tif")
        assert isinstance(r, DefaultResolver)

    def test_auto_discovers_earthdaily_resolver(self):
        resolver_registry._resolvers = []
        r = get_resolver_for_url("https://api.earthdaily.com/file.tif")
        assert isinstance(r, EarthDailyAPIResolver)

    def test_auto_discovered_resolver_is_cached(self):
        resolver_registry._resolvers = []
        r1 = get_resolver_for_url("https://api.earthdaily.com/file.tif")
        r2 = get_resolver_for_url("https://stac.earthdaily.com/file.tif")
        assert r1 is r2

    def test_configured_instance_takes_priority(self):
        resolver_registry._resolvers = []
        configured = EarthDailyAPIResolver(api_key="configured-key")
        resolver_registry.register(configured)
        r = get_resolver_for_url("https://api.earthdaily.com/file.tif")
        assert r is configured
        assert r.api_key == "configured-key"

    def test_configured_earthdaily_resolver_sets_key_from_requester(self):
        resolver_registry._resolvers = []
        configured = EarthDailyAPIResolver(api_key="old-key")
        resolver_registry.register(configured)

        mock_requester = MagicMock()
        mock_requester.auth.get_token.return_value = "fresh-token"

        r = get_resolver_for_url("https://api.earthdaily.com/file.tif", api_requester=mock_requester)
        assert r is configured
        assert r.api_key == "fresh-token"

    def test_configured_earthdaily_resolver_no_auth_sets_none(self):
        resolver_registry._resolvers = []
        configured = EarthDailyAPIResolver(api_key="old-key")
        resolver_registry.register(configured)

        mock_requester = MagicMock()
        mock_requester.auth = None

        r = get_resolver_for_url("https://api.earthdaily.com/file.tif", api_requester=mock_requester)
        assert r is configured
        assert r.api_key is None

    def test_earthdaily_resolver_gets_api_key_from_requester(self):
        resolver_registry._resolvers = []
        mock_requester = MagicMock()
        mock_requester.auth.get_token.return_value = "fresh-token"
        mock_requester.config = SimpleNamespace()

        r = get_resolver_for_url("https://api.earthdaily.com/file.tif", api_requester=mock_requester)
        assert isinstance(r, EarthDailyAPIResolver)
        assert r.api_key == "fresh-token"

    def test_needs_credentials_raises_without_config(self):
        resolver_registry._resolvers = []
        url = "https://data.lpdaac.earthdatacloud.nasa.gov/file.hdf"
        with self.assertRaises(ResolverNotConfiguredError) as ctx:
            get_resolver_for_url(url)
        assert "EarthDataResolver" in str(ctx.exception)

    def test_needs_credentials_uses_from_config(self):
        resolver_registry._resolvers = []
        from earthdaily.platform._download_resolvers import EarthDataResolver

        mock_requester = MagicMock()
        mock_requester.config = SimpleNamespace(earthdata_token="my-jwt-token")

        url = "https://data.lpdaac.earthdatacloud.nasa.gov/file.hdf"
        r = get_resolver_for_url(url, api_requester=mock_requester)
        assert isinstance(r, EarthDataResolver)

    def test_needs_credentials_raises_when_config_empty(self):
        resolver_registry._resolvers = []
        mock_requester = MagicMock()
        mock_requester.config = SimpleNamespace(earthdata_token="")

        url = "https://data.lpdaac.earthdatacloud.nasa.gov/file.hdf"
        with self.assertRaises(ResolverNotConfiguredError):
            get_resolver_for_url(url, api_requester=mock_requester)

    def test_s3_auto_discovered_without_credentials(self):
        resolver_registry._resolvers = []
        from earthdaily.platform._download_resolvers import S3Resolver

        r = get_resolver_for_url("s3://my-bucket/path/file.tif")
        assert isinstance(r, S3Resolver)

    def test_resolver_classes_tuple_contains_expected(self):
        from earthdaily.platform._download_resolvers import EarthDataResolver, EUMETSATResolver, S3Resolver

        assert EarthDailyAPIResolver in RESOLVER_CLASSES
        assert EarthDataResolver in RESOLVER_CLASSES
        assert EUMETSATResolver in RESOLVER_CLASSES
        assert S3Resolver in RESOLVER_CLASSES


if __name__ == "__main__":
    unittest.main()
