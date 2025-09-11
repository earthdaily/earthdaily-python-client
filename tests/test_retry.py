import unittest
from unittest.mock import Mock, patch

from earthdaily._api_requester import APIRequester
from earthdaily._eds_config import EDSConfig
from earthdaily._http_client import HTTPClient
from earthdaily.platform import PlatformService


class TestRetryFunctionality(unittest.TestCase):
    def test_api_requester_uses_retry_config(self):
        config = EDSConfig(
            client_id="test_id",
            client_secret="test_secret",
            token_url="https://test.com/token",
            max_retries=5,
            retry_backoff_factor=2.0,
            bypass_auth=True,
        )

        api_requester = APIRequester(config=config)

        self.assertEqual(api_requester.http_client.max_retries, 5)
        self.assertEqual(api_requester.http_client.retry_backoff_factor, 2.0)

    @patch("earthdaily.platform.Client.open")
    def test_platform_service_retry_session(self, mock_client_open):
        mock_client = Mock()
        mock_client_open.return_value = mock_client

        config = EDSConfig(
            client_id="test_id",
            client_secret="test_secret",
            token_url="https://test.com/token",
            max_retries=4,
            retry_backoff_factor=1.5,
            bypass_auth=True,
        )

        api_requester = APIRequester(config=config)
        PlatformService(api_requester, config.asset_access_mode)

        # Verify StacApiIO is configured and passed to Client.open
        call_args = mock_client_open.call_args
        stac_io = call_args[1]["stac_io"]
        self.assertEqual(stac_io.__class__.__name__, "StacApiIO")
        self.assertIsNotNone(stac_io.session)

    def test_http_client_session_reuse(self):
        client = HTTPClient(max_retries=3, retry_backoff_factor=1.0)

        # First call creates session
        session1 = client._get_session()
        # Second call reuses same session
        session2 = client._get_session()

        self.assertIs(session1, session2)

    @patch("earthdaily._http_client.requests.Session")
    @patch("earthdaily._http_client.HTTPAdapter")
    @patch("earthdaily._http_client.Retry")
    def test_retry_strategy_configuration(self, mock_retry, mock_adapter, mock_session_class):
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        client = HTTPClient(max_retries=3, retry_backoff_factor=1.5)
        client._get_session()

        # Verify Retry is configured correctly
        mock_retry.assert_called_once_with(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1.5,
        )

        # Verify HTTPAdapter is created with retry strategy
        mock_adapter.assert_called_once_with(max_retries=mock_retry.return_value)

        # Verify session mounts are configured
        mock_session.mount.assert_any_call("https://", mock_adapter.return_value)
        mock_session.mount.assert_any_call("http://", mock_adapter.return_value)


if __name__ == "__main__":
    unittest.main()
