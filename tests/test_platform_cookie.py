import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

from requests.structures import CaseInsensitiveDict

from earthdaily._eds_config import AssetAccessMode, EDSConfig
from earthdaily._http_client import HTTPResponse
from earthdaily._platform_cookie import (
    COOKIE_DOMAIN,
    COOKIE_NAME,
    COOKIE_PATH,
    NETSCAPE_HEADER,
    PlatformAssetCookieClient,
    PlatformCookieManager,
)
from earthdaily.exceptions import EDSAPIError


def _proxy_config(
    platform_cookie_enabled: bool = True,
    platform_cookie_auto_refresh: bool = False,
    asset_access_mode: AssetAccessMode = AssetAccessMode.PROXY_URLS,
) -> EDSConfig:
    return EDSConfig(
        bypass_auth=True,
        asset_access_mode=asset_access_mode,
        platform_cookie_enabled=platform_cookie_enabled,
        platform_cookie_auto_refresh=platform_cookie_auto_refresh,
    )


def _mock_api_requester(base_url: str = "https://api.example.com") -> Mock:
    api_requester = Mock()
    api_requester.base_url = base_url
    return api_requester


def _cookie_response(value: str = "abc123", max_age: int = 3600, status_code: int = 204) -> HTTPResponse:
    headers: CaseInsensitiveDict[str] = CaseInsensitiveDict(
        {"Set-Cookie": f"{COOKIE_NAME}={value}; Max-Age={max_age}; Path={COOKIE_PATH}"}
    )
    return HTTPResponse(status_code=status_code, body={}, headers=headers)


class TestPlatformAssetCookieClient(unittest.TestCase):
    def test_fetch_sends_post_with_max_age(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response()

        client = PlatformAssetCookieClient(api_requester)
        token = client.fetch(ttl_seconds=7200)

        api_requester.send_request.assert_called_once()
        request = api_requester.send_request.call_args[0][0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.url, "https://api.example.com/platform/v1/asset/cookie")
        self.assertEqual(request.body, {"max_age": 7200})

        self.assertEqual(token.value, "abc123")
        self.assertGreater(token.expires_at, time.time())

    def test_fetch_raises_on_error_status(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = HTTPResponse(500, {"error": "boom"}, CaseInsensitiveDict())

        client = PlatformAssetCookieClient(api_requester)
        with self.assertRaises(EDSAPIError):
            client.fetch(ttl_seconds=3600)

    def test_fetch_raises_when_set_cookie_missing(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = HTTPResponse(204, {}, CaseInsensitiveDict())

        client = PlatformAssetCookieClient(api_requester)
        with self.assertRaises(EDSAPIError):
            client.fetch(ttl_seconds=3600)

    def test_fetch_raises_when_platform_asset_cookie_missing(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = HTTPResponse(
            204, {}, CaseInsensitiveDict({"Set-Cookie": "OTHER=x; Max-Age=60"})
        )

        client = PlatformAssetCookieClient(api_requester)
        with self.assertRaises(EDSAPIError):
            client.fetch(ttl_seconds=3600)


class TestPlatformCookieManagerValidation(unittest.TestCase):
    def test_raises_when_feature_disabled(self):
        config = _proxy_config(platform_cookie_enabled=False)
        manager = PlatformCookieManager(_mock_api_requester(), config)

        with self.assertRaises(ValueError):
            manager.get_cookie_file(path="/tmp/should-not-be-used.txt")

    def test_raises_when_not_proxy_mode(self):
        config = _proxy_config(asset_access_mode=AssetAccessMode.PRESIGNED_URLS)
        manager = PlatformCookieManager(_mock_api_requester(), config)

        with self.assertRaises(ValueError):
            manager.get_cookie_file(path="/tmp/should-not-be-used.txt")

    def test_raises_when_path_is_none(self):
        manager = PlatformCookieManager(_mock_api_requester(), _proxy_config())

        with self.assertRaises(ValueError):
            manager.get_cookie_file(path=None)  # type: ignore[arg-type]

    def test_raises_when_parent_directory_missing(self):
        manager = PlatformCookieManager(_mock_api_requester(), _proxy_config())
        missing_parent = Path(tempfile.gettempdir()) / "earthdaily-missing-dir-xyz" / "cookies.txt"

        with self.assertRaises(ValueError):
            manager.get_cookie_file(path=missing_parent)


class TestPlatformCookieManagerWrite(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_tmp)
        self.path = Path(self.tmpdir) / "cookies.txt"

    def _cleanup_tmp(self):
        for name in os.listdir(self.tmpdir):
            try:
                os.remove(os.path.join(self.tmpdir, name))
            except FileNotFoundError:
                pass
        os.rmdir(self.tmpdir)

    def test_preserves_existing_cookies_in_file(self):
        nasa_line = ".nasa.gov\tTRUE\t/\tFALSE\t9999999999\tNASA_TOKEN\tnasavalue\n"
        self.path.write_text(NETSCAPE_HEADER + nasa_line)

        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response(value="edtoken")

        manager = PlatformCookieManager(api_requester, _proxy_config())
        manager.get_cookie_file(path=self.path)

        contents = self.path.read_text()
        self.assertIn("NASA_TOKEN", contents)
        self.assertIn("edtoken", contents)

    def test_refresh_does_not_duplicate_cookie_entry(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response(value="tok1", max_age=3600)

        manager = PlatformCookieManager(api_requester, _proxy_config())
        manager.get_cookie_file(path=self.path)
        manager.get_cookie_file(path=self.path, force_refresh=True)

        contents = self.path.read_text()
        self.assertEqual(contents.count(COOKIE_NAME), 1)

    def test_writes_netscape_file_with_expected_contents(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response(value="tok1", max_age=1800)

        manager = PlatformCookieManager(api_requester, _proxy_config())
        returned = manager.get_cookie_file(path=self.path, ttl_seconds=1800)

        self.assertEqual(returned, self.path.resolve())
        contents = self.path.read_text()
        self.assertTrue(contents.startswith(NETSCAPE_HEADER))
        self.assertIn(COOKIE_DOMAIN, contents)
        self.assertIn(COOKIE_PATH, contents)
        self.assertIn(COOKIE_NAME, contents)
        self.assertIn("tok1", contents)

    def test_file_has_restrictive_permissions(self):
        if os.name == "nt":
            self.skipTest("POSIX permissions not applicable on Windows")

        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response()

        manager = PlatformCookieManager(api_requester, _proxy_config())
        manager.get_cookie_file(path=self.path)

        mode = os.stat(self.path).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_ttl_seconds_is_passed_to_endpoint(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response()

        manager = PlatformCookieManager(api_requester, _proxy_config())
        manager.get_cookie_file(path=self.path, ttl_seconds=9000)

        request = api_requester.send_request.call_args[0][0]
        self.assertEqual(request.body, {"max_age": 9000})

    def test_does_not_refetch_when_token_still_valid(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response(max_age=3600)

        manager = PlatformCookieManager(api_requester, _proxy_config())
        manager.get_cookie_file(path=self.path)
        manager.get_cookie_file(path=self.path)

        self.assertEqual(api_requester.send_request.call_count, 1)

    def test_force_refresh_triggers_new_fetch(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response(max_age=3600)

        manager = PlatformCookieManager(api_requester, _proxy_config())
        manager.get_cookie_file(path=self.path)
        manager.get_cookie_file(path=self.path, force_refresh=True)

        self.assertEqual(api_requester.send_request.call_count, 2)

    def test_atomic_write_does_not_leave_temp_files(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response()

        manager = PlatformCookieManager(api_requester, _proxy_config())
        manager.get_cookie_file(path=self.path)

        leftovers = [name for name in os.listdir(self.tmpdir) if name.startswith(".platform-cookies-")]
        self.assertEqual(leftovers, [])

    def test_close_leaves_cookie_file(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response()

        manager = PlatformCookieManager(api_requester, _proxy_config())
        manager.get_cookie_file(path=self.path)
        self.assertTrue(self.path.exists())

        manager.close()
        self.assertTrue(self.path.exists())


class TestPlatformCookieManagerAutoRefresh(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_tmp)
        self.path = Path(self.tmpdir) / "cookies.txt"

    def _cleanup_tmp(self):
        for name in os.listdir(self.tmpdir):
            try:
                os.remove(os.path.join(self.tmpdir, name))
            except FileNotFoundError:
                pass
        os.rmdir(self.tmpdir)

    def test_auto_refresh_thread_does_not_start_when_disabled(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response()

        config = _proxy_config(platform_cookie_auto_refresh=False)
        manager = PlatformCookieManager(api_requester, config)
        manager.get_cookie_file(path=self.path)

        self.assertIsNone(manager._thread)  # type: ignore[attr-defined]

    def test_auto_refresh_thread_starts_when_enabled(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response()

        config = _proxy_config(platform_cookie_auto_refresh=True)
        manager = PlatformCookieManager(api_requester, config)
        manager.get_cookie_file(path=self.path)

        try:
            thread = manager._thread  # type: ignore[attr-defined]
            self.assertIsNotNone(thread)
            assert thread is not None
            self.assertTrue(thread.daemon)
            self.assertTrue(thread.is_alive())
        finally:
            manager.close()

    def test_stop_auto_refresh_joins_thread(self):
        api_requester = _mock_api_requester()
        api_requester.send_request.return_value = _cookie_response()

        config = _proxy_config(platform_cookie_auto_refresh=True)
        manager = PlatformCookieManager(api_requester, config)
        manager.get_cookie_file(path=self.path)

        thread = manager._thread  # type: ignore[attr-defined]
        self.assertIsNotNone(thread)
        manager.stop_auto_refresh()
        assert thread is not None
        self.assertFalse(thread.is_alive())

    def test_start_auto_refresh_requires_path(self):
        manager = PlatformCookieManager(_mock_api_requester(), _proxy_config())
        with self.assertRaises(ValueError):
            manager.start_auto_refresh()


class TestEDSClientCookieMethods(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_tmp)
        self.path = Path(self.tmpdir) / "cookies.txt"

    def _cleanup_tmp(self):
        for name in os.listdir(self.tmpdir):
            try:
                os.remove(os.path.join(self.tmpdir, name))
            except FileNotFoundError:
                pass
        os.rmdir(self.tmpdir)

    def _make_client(self, **config_overrides):
        from earthdaily._eds_client import EDSClient

        config = _proxy_config(**config_overrides)
        client = EDSClient(config)
        client.api_requester = _mock_api_requester()
        client.api_requester.send_request = MagicMock(return_value=_cookie_response())
        return client

    def test_get_platform_cookie_file_writes_file(self):
        client = self._make_client()
        returned = client.get_platform_cookie_file(path=self.path)

        self.assertEqual(returned, self.path.resolve())
        self.assertTrue(self.path.exists())

    def test_get_platform_cookie_raises_when_feature_disabled(self):
        from earthdaily._eds_client import EDSClient

        config = EDSConfig(bypass_auth=True, asset_access_mode=AssetAccessMode.PROXY_URLS)
        client = EDSClient(config)

        with self.assertRaises(ValueError):
            client.get_platform_cookie_file(path=self.path)

    def test_get_platform_cookie_raises_when_mode_not_proxy(self):
        from earthdaily._eds_client import EDSClient

        config = EDSConfig(
            bypass_auth=True,
            asset_access_mode=AssetAccessMode.PRESIGNED_URLS,
            platform_cookie_enabled=True,
        )
        client = EDSClient(config)

        with self.assertRaises(ValueError):
            client.get_platform_cookie_file(path=self.path)

    def test_close_leaves_cookie_file(self):
        client = self._make_client()
        client.get_platform_cookie_file(path=self.path)
        self.assertTrue(self.path.exists())

        client.close()
        self.assertTrue(self.path.exists())

    def test_context_manager_leaves_cookie_file(self):
        from earthdaily._eds_client import EDSClient

        config = _proxy_config()
        with EDSClient(config) as client:
            client.api_requester = _mock_api_requester()
            client.api_requester.send_request = MagicMock(return_value=_cookie_response())
            client.get_platform_cookie_file(path=self.path)
            self.assertTrue(self.path.exists())

        self.assertTrue(self.path.exists())


class TestEDSConfigCookieDefaults(unittest.TestCase):
    def test_defaults_are_disabled(self):
        config = EDSConfig(bypass_auth=True)
        self.assertFalse(config.platform_cookie_enabled)
        self.assertFalse(config.platform_cookie_auto_refresh)


if __name__ == "__main__":
    unittest.main()
