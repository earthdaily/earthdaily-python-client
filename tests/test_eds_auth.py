import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from earthdaily.earthdatastore import Auth, EarthDataStoreConfig


class TestEdsAuth(unittest.TestCase):
    def setUp(self):
        self.auth_instance = Auth()

    @patch("earthdaily.earthdatastore.Auth.read_credentials_from_json")
    def test_read_credentials_from_json(self, mock_read_json):
        mock_read_json.return_value = {
            "EDS_AUTH_URL": "https://auth.example.com",
            "EDS_SECRET": "secret_value",
            "EDS_CLIENT_ID": "client_id_value",
        }
        json_path = Path("/path/to/credentials.json")
        credentials = Auth.read_credentials_from_json(json_path)
        mock_read_json.assert_called_once_with(json_path)
        self.assertEqual(credentials["EDS_AUTH_URL"], "https://auth.example.com")

    @patch("earthdaily.earthdatastore.Auth.read_credentials_from_toml")
    def test_read_credentials_from_toml(self, mock_read_toml):
        mock_read_toml.return_value = {
            "EDS_AUTH_URL": "https://auth.example.com",
            "EDS_SECRET": "secret_value",
            "EDS_CLIENT_ID": "client_id_value",
        }
        toml_path = Path("/path/to/credentials.toml")
        credentials = Auth.read_credentials_from_toml(toml_path, profile="default")
        mock_read_toml.assert_called_once_with(toml_path, profile="default")
        self.assertEqual(credentials["EDS_SECRET"], "secret_value")

    @patch("os.getenv")
    def test_read_credentials_from_environment(self, mock_getenv):
        mock_getenv.side_effect = lambda key: {
            "EDS_AUTH_URL": "https://auth.example.com",
            "EDS_SECRET": "secret_value",
            "EDS_CLIENT_ID": "client_id_value",
        }.get(key)
        credentials = Auth.read_credentials_from_environment()
        self.assertEqual(credentials["EDS_CLIENT_ID"], "client_id_value")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="[default]\nEDS_AUTH_URL=https://auth.example.com\nEDS_SECRET=secret_value\nEDS_CLIENT_ID=client_id_value",
    )
    @patch("pathlib.Path.exists", return_value=True)
    def test_read_credentials_from_ini(self, mock_exists, mock_open):
        credentials = Auth.read_credentials_from_ini(profile="default")
        self.assertEqual(credentials["EDS_AUTH_URL"], "https://auth.example.com")
        self.assertEqual(credentials["EDS_SECRET"], "secret_value")
        self.assertEqual(credentials["EDS_CLIENT_ID"], "client_id_value")

    def test_parse_dict(self):
        config = {
            "EDS_AUTH_URL": "https://auth.example.com",
            "EDS_SECRET": "secret_value",
            "EDS_CLIENT_ID": "client_id_value",
        }
        result = self.auth_instance._config_parser(config=config)
        self.assertIsInstance(result, EarthDataStoreConfig)
        self.assertEqual(result.auth_url, "https://auth.example.com")
        self.assertEqual(result.client_secret, "secret_value")
        self.assertEqual(result.client_id, "client_id_value")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps(
            {
                "EDS_AUTH_URL": "https://auth.example.com",
                "EDS_SECRET": "secret_value",
                "EDS_CLIENT_ID": "client_id_value",
            }
        ),
    )
    def test_parse_json_file(self, mock_open_file):
        result = self.auth_instance._config_parser(config="path/to/credentials.json")
        self.assertIsInstance(result, EarthDataStoreConfig)
        self.assertEqual(result.auth_url, "https://auth.example.com")
        self.assertEqual(result.client_secret, "secret_value")
        self.assertEqual(result.client_id, "client_id_value")
        mock_open_file.assert_called_once_with("path/to/credentials.json", "rb")

    def test_parse_tuple(self):
        config = ("mock_access_token", "https://api.example.com")
        result = self.auth_instance._config_parser(config=config)
        self.assertIsInstance(result, EarthDataStoreConfig)
        self.assertEqual(result.access_token, "mock_access_token")
        self.assertEqual(result.eds_url, "https://api.example.com")

    @patch("os.getenv")
    def test_parse_from_environment(self, mock_getenv):
        mock_getenv.side_effect = lambda key, default=None: {
            "EDS_AUTH_URL": "https://auth.example.com",
            "EDS_SECRET": "secret_value",
            "EDS_CLIENT_ID": "client_id_value",
            "EDS_API_URL": "https://api.example.com",
        }.get(key, default)
        result = self.auth_instance._config_parser()
        self.assertIsInstance(result, EarthDataStoreConfig)
        self.assertEqual(result.auth_url, "https://auth.example.com")
        self.assertEqual(result.client_secret, "secret_value")
        self.assertEqual(result.client_id, "client_id_value")
        self.assertEqual(result.eds_url, "https://api.example.com")

    def test_missing_credentials(self):
        with self.assertRaises(AttributeError) as context:
            self.auth_instance._config_parser(config={})
        self.assertIn(
            "You need to have env : EDS_AUTH_URL, EDS_SECRET and EDS_CLIENT_ID",
            str(context.exception),
        )

    @patch("requests.post")
    def test_get_access_token(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "mock_access_token"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        config = EarthDataStoreConfig(
            auth_url="https://auth.example.com",
            client_secret="secret_value",
            client_id="client_id_value",
        )

        token = self.auth_instance.get_access_token(config=config)

        mock_post.assert_called_once_with(
            "https://auth.example.com",
            data={"grant_type": "client_credentials"},
            allow_redirects=False,
            auth=("client_id_value", "secret_value"),
        )
        self.assertEqual(token, "mock_access_token")

    @patch("earthdaily.earthdatastore.Auth._get_client")
    @patch("earthdaily.earthdatastore.Auth.read_credentials")
    def test_from_credentials(self, mock_read_credentials, mock_get_client):
        mock_read_credentials.return_value = {
            "EDS_AUTH_URL": "https://auth.example.com",
            "EDS_SECRET": "secret_value",
            "EDS_CLIENT_ID": "client_id_value",
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "mock_access_token"}
        mock_response.raise_for_status = MagicMock()

        mock_get_client.return_value = MagicMock()

        auth_instance = Auth.from_credentials(
            json_path=Path("/path/to/credentials.json")
        )

        self.assertIsInstance(auth_instance, Auth)
        self.assertEqual(
            auth_instance._Auth__auth_config["EDS_CLIENT_ID"], "client_id_value"
        )
        self.assertEqual(auth_instance._Auth__auth_config["EDS_SECRET"], "secret_value")
        self.assertEqual(
            auth_instance._Auth__auth_config["EDS_AUTH_URL"], "https://auth.example.com"
        )

        mock_get_client.assert_called_once()


if __name__ == "__main__":
    unittest.main()
