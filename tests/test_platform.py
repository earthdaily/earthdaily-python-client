import unittest
from unittest.mock import MagicMock, patch

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
            token_url="https://token_url",  # Fake URL but will be mocked
            base_url="https://example.com",
            pre_sign_urls=True,
        )
        auth = Authentication(config.client_id, config.client_secret, config.token_url)
        api_requester = APIRequester(base_url=config.base_url, auth=auth)
        platform_service = PlatformService(api_requester, config.pre_sign_urls)

        # Test pystac_client.Client.open
        mock_client_open.assert_called_once_with(
            "https://example.com/platform/v1/stac",
            stac_io=mock_client_open.call_args[1]["stac_io"],
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer mock_token",
                "X-Signed-Asset-Urls": "True",
            },
        )

        self.assertEqual(platform_service.pystac_client, mock_client)


if __name__ == "__main__":
    unittest.main()
