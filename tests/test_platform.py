import unittest
from unittest.mock import ANY, MagicMock, patch

from pystac_client.stac_api_io import StacApiIO

from earthdaily._api_requester import APIRequester
from earthdaily._auth_client import Authentication
from earthdaily._eds_config import EDSConfig
from earthdaily.platform import PlatformService


class TestPlatformService(unittest.TestCase):
    @patch("earthdaily.platform.Client.open")
    @patch("earthdaily._auth_client.Authentication.get_token", return_value="mock_token")
    @patch("earthdaily._auth_client.requests.Session.post")
    def test_platform_service_initialization(self, mock_post, mock_get_token, mock_client_open):
        # Mock the auth HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "mock_token", "expires_in": 3600}
        mock_post.return_value = mock_response

        # Mock Client.open return value to prevent real API call
        mock_client = MagicMock()
        mock_client_open.return_value = mock_client

        # Setup
        config = EDSConfig(
            client_id="client_id",
            client_secret="client_secret",
            token_url="https://token_url",
            base_url="https://example.com",
            asset_access_mode="presigned-urls",
        )
        auth = Authentication(config.client_id, config.client_secret, config.token_url)
        api_requester = APIRequester(config=config, auth=auth)
        platform_service = PlatformService(api_requester, config.asset_access_mode)

        # Test pystac_client.Client.open
        mock_client_open.assert_called_once_with(
            "https://example.com/platform/v1/stac",
            stac_io=mock_client_open.call_args[1]["stac_io"],
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer mock_token",
                "X-Signed-Asset-Urls": "True",
                "User-Agent": ANY,
                "X-EDA-Client-User-Agent": ANY,
            },
        )

        # Verify StacApiIO is passed to Client.open
        stac_io = mock_client_open.call_args[1]["stac_io"]
        self.assertIsInstance(stac_io, StacApiIO)
        self.assertIsNotNone(stac_io.session)

        self.assertEqual(platform_service.pystac_client, mock_client)


if __name__ == "__main__":
    unittest.main()
