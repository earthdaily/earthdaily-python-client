import unittest
from unittest.mock import patch

from earthdatastore._auth_client import CognitoAuth
from earthdatastore._eds_client import EDSClient
from earthdatastore._eds_config import EDSConfig


class TestEDSClient(unittest.TestCase):
    @patch("earthdatastore._auth_client.CognitoAuth.get_token", return_value="test_token")
    def test_create_auth_cognito(self, mock_auth):
        config = EDSConfig(
            auth_method="cognito",
            client_id="client_id",
            client_secret="client_secret",
            token_url="token_url",
        )
        client = EDSClient(config)

        self.assertIsInstance(client.auth, CognitoAuth)
        self.assertEqual(client.api_requester.auth.get_token(), "test_token")

    def test_unsupported_auth_method(self):
        config = EDSConfig(
            auth_method="unsupported_method",
            client_id="client_id",
            client_secret="client_secret",
            token_url="token_url",
        )

        with self.assertRaises(ValueError) as context:
            EDSClient(config)

        self.assertIn("Unsupported auth method", str(context.exception))


if __name__ == "__main__":
    unittest.main()
