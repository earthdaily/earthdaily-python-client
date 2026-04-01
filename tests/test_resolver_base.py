import re
import unittest
from pathlib import Path

from earthdaily.platform._resolver_base import (
    DefaultResolver,
    DomainPatternResolver,
    EarthDailyAPIResolver,
)


class TestDefaultResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = DefaultResolver()

    def test_handles_http(self):
        assert self.resolver.can_handle("http://example.com/file.tif")

    def test_handles_https(self):
        assert self.resolver.can_handle("https://example.com/file.tif")

    def test_rejects_ftp(self):
        assert not self.resolver.can_handle("ftp://example.com/file.tif")

    def test_rejects_s3(self):
        assert not self.resolver.can_handle("s3://bucket/file.tif")

    def test_get_download_url_passthrough(self):
        url = "https://example.com/file.tif"
        assert self.resolver.get_download_url(url) == url

    def test_get_headers_empty(self):
        assert self.resolver.get_headers("https://example.com/file.tif") == {}

    def test_download_returns_none(self):
        assert self.resolver.download("https://example.com/f.tif", Path("/tmp")) is None

    def test_download_quiet_returns_none(self):
        assert self.resolver.download("https://example.com/f.tif", Path("/tmp"), quiet=True) is None


class TestDomainPatternResolver(unittest.TestCase):
    def test_no_pattern_rejects_all(self):
        class Bare(DomainPatternResolver):
            def get_download_url(self, url):
                return url

            def get_headers(self, url):
                return {}

        assert not Bare().can_handle("https://example.com/file.tif")

    def test_pattern_matches(self):
        class MyResolver(DomainPatternResolver):
            domain_pattern = re.compile(r".*example\.com$")

            def get_download_url(self, url):
                return url

            def get_headers(self, url):
                return {}

        r = MyResolver()
        assert r.can_handle("https://example.com/file.tif")
        assert r.can_handle("http://sub.example.com/file.tif")
        assert not r.can_handle("https://other-domain.com/file.tif")


class TestEarthDailyAPIResolver(unittest.TestCase):
    def test_handles_earthdaily_domains(self):
        r = EarthDailyAPIResolver()
        assert r.can_handle("https://api.earthdaily.com/file.tif")
        assert r.can_handle("https://stac.earthdaily.com/file.tif")

    def test_rejects_other_domains(self):
        r = EarthDailyAPIResolver()
        assert not r.can_handle("https://example.com/file.tif")

    def test_no_key_returns_empty_headers(self):
        r = EarthDailyAPIResolver()
        assert r.get_headers("https://api.earthdaily.com/f") == {}

    def test_key_returns_bearer_header(self):
        r = EarthDailyAPIResolver(api_key="test-key")
        headers = r.get_headers("https://api.earthdaily.com/f")
        assert headers == {"Authorization": "Bearer test-key"}

    def test_get_download_url_passthrough(self):
        r = EarthDailyAPIResolver()
        url = "https://api.earthdaily.com/file.tif"
        assert r.get_download_url(url) == url


if __name__ == "__main__":
    unittest.main()
