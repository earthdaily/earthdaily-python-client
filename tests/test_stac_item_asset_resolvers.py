import re
import unittest
from unittest.mock import patch
from urllib.parse import urlparse

from earthdaily.platform._stac_item_asset import (
    AssetResolver,
    DefaultResolver,
    DomainPatternResolver,
    EarthDailyAPIResolver,
    ResolverRegistry,
    get_resolver_for_url,
)


class TestAssetResolvers(unittest.TestCase):
    def test_default_resolver(self):
        """Test DefaultResolver functionality"""
        resolver = DefaultResolver()

        self.assertTrue(resolver.can_handle("http://example.com/file.tif"))
        self.assertTrue(resolver.can_handle("https://example.com/file.tif"))
        self.assertFalse(resolver.can_handle("ftp://example.com/file.tif"))
        self.assertFalse(resolver.can_handle("s3://bucket/file.tif"))

        url = "https://example.com/file.tif"
        self.assertEqual(resolver.get_download_url(url), url)
        self.assertEqual(resolver.get_headers(url), {})

    def test_domain_pattern_resolver(self):
        """Test DomainPatternResolver functionality"""

        class TestBaseResolver(DomainPatternResolver):
            def get_download_url(self, url):
                return url

            def get_headers(self, url):
                return {}

        base_resolver = TestBaseResolver()
        self.assertFalse(base_resolver.can_handle("https://example.com/file.tif"))

        class TestResolver(DomainPatternResolver):
            domain_pattern = re.compile(r".*example\.com$")

            def get_download_url(self, url):
                return url

            def get_headers(self, url):
                return {}

        test_resolver = TestResolver()
        self.assertTrue(test_resolver.can_handle("https://example.com/file.tif"))
        self.assertTrue(test_resolver.can_handle("http://sub.example.com/file.tif"))
        self.assertFalse(test_resolver.can_handle("https://another-domain.com/file.tif"))

    def test_earthdaily_api_resolver(self):
        """Test EarthDailyAPIResolver functionality"""
        resolver_no_key = EarthDailyAPIResolver()

        self.assertTrue(resolver_no_key.can_handle("https://api.earthdaily.com/file.tif"))
        self.assertTrue(resolver_no_key.can_handle("https://stac.earthdaily.com/file.tif"))
        self.assertFalse(resolver_no_key.can_handle("https://example.com/file.tif"))

        url = "https://api.earthdaily.com/file.tif"
        self.assertEqual(resolver_no_key.get_download_url(url), url)
        self.assertEqual(resolver_no_key.get_headers(url), {})

        resolver_with_key = EarthDailyAPIResolver(api_key="test-api-key")
        headers = resolver_with_key.get_headers(url)
        self.assertEqual(headers, {"Authorization": "Bearer test-api-key"})

    def test_resolver_registry(self):
        """Test ResolverRegistry functionality"""
        registry = ResolverRegistry()

        mock_resolver1 = type(
            "MockResolver1",
            (AssetResolver,),
            {
                "can_handle": lambda self, url: urlparse(url).netloc == "domain1.com",
                "get_download_url": lambda self, url: f"modified-{url}",
                "get_headers": lambda self, url: {"X-Mock1": "Value1"},
            },
        )()

        mock_resolver2 = type(
            "MockResolver2",
            (AssetResolver,),
            {
                "can_handle": lambda self, url: urlparse(url).netloc == "domain2.com",
                "get_download_url": lambda self, url: f"modified-{url}",
                "get_headers": lambda self, url: {"X-Mock2": "Value2"},
            },
        )()

        registry.register(mock_resolver1)
        registry.register(mock_resolver2)

        resolver1 = registry.get_resolver("https://domain1.com/file.tif")
        self.assertEqual(resolver1, mock_resolver1)

        resolver2 = registry.get_resolver("https://domain2.com/file.tif")
        self.assertEqual(resolver2, mock_resolver2)

        resolver3 = registry.get_resolver("https://unknown.com/file.tif")
        self.assertIsInstance(resolver3, DefaultResolver)

    @patch("earthdaily.platform._stac_item_asset.resolver_registry.get_resolver")
    def test_get_resolver_for_url(self, mock_get_resolver):
        """Test get_resolver_for_url function"""
        mock_resolver = DefaultResolver()
        mock_get_resolver.return_value = mock_resolver

        result = get_resolver_for_url("https://example.com/file.tif")

        self.assertEqual(result, mock_resolver)
        mock_get_resolver.assert_called_once_with("https://example.com/file.tif")


if __name__ == "__main__":
    unittest.main()
