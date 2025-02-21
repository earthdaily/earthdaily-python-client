import unittest
from unittest.mock import patch

from earthdaily._auth_client import Authentication
from earthdaily._eds_client import EDSClient
from earthdaily._eds_config import EDSConfig


class TestEDSClient(unittest.TestCase):
    @patch("earthdaily._auth_client.Authentication.get_token", return_value="test_token")
    def test_create_auth(self, mock_auth):
        config = EDSConfig(
            client_id="client_id",
            client_secret="client_secret",
            token_url="token_url",
        )
        client = EDSClient(config)

        self.assertIsInstance(client.auth, Authentication)
        self.assertEqual(client.api_requester.auth.get_token(), "test_token")


if __name__ == "__main__":
    unittest.main()
