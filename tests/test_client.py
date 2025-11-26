import unittest
from unittest.mock import patch

from earthdaily._auth_client import Authentication
from earthdaily._eds_client import EDSClient
from earthdaily._eds_config import EDSConfig


class TestEDSClient(unittest.TestCase):
    @patch("earthdaily._auth_client.Authentication.authenticate")
    @patch("earthdaily._auth_client.Authentication.get_token", return_value="test_token")
    def test_create_auth(self, mock_get_token, mock_authenticate):
        config = EDSConfig(
            client_id="client_id",
            client_secret="client_secret",
            token_url="token_url",
        )
        client = EDSClient(config)

        self.assertIsInstance(client.auth, Authentication)
        mock_authenticate.assert_called_once()
        self.assertEqual(client.api_requester.auth.get_token(), "test_token")

    def test_create_client_with_bypass_auth(self):
        config = EDSConfig(bypass_auth=True, base_url="https://api.earthdaily.com")
        client = EDSClient(config)

        # Auth should be None when bypassed
        self.assertIsNone(client.auth)
        # Client should still be created successfully
        self.assertIsNotNone(client)
        self.assertEqual(client.api_requester.base_url, "https://api.earthdaily.com")

    def test_bypass_auth_no_credentials_required(self):
        config = EDSConfig(bypass_auth=True)
        client = EDSClient(config)

        self.assertIsNone(client.auth)
        self.assertIsNotNone(client.api_requester)


if __name__ == "__main__":
    unittest.main()
