import unittest
from unittest.mock import ANY, MagicMock, patch

from earthdaily import EarthDataStore
from earthdaily.earthdatastore import Auth


class TestEarthDataStore(unittest.TestCase):
    @patch("earthdaily.earthdatastore.Auth.from_credentials")
    def test_asset_proxy_enabled(self, mock_from_credentials):
        # Mock the return value of from_credentials
        mock_auth_instance = MagicMock(spec=Auth)
        mock_from_credentials.return_value = mock_auth_instance

        # Call EarthDataStore with asset_proxy_enabled set to True
        auth_instance = EarthDataStore(asset_proxy_enabled=True)

        # Assert that from_credentials was called with asset_proxy_enabled=True
        mock_from_credentials.assert_called_once_with(
            json_path=None,
            toml_path=None,
            profile=None,
            client_version=ANY,
            presign_urls=True,
            asset_proxy_enabled=True,
        )

        # Assert that the returned instance is the mocked instance
        self.assertEqual(auth_instance, mock_auth_instance)


if __name__ == "__main__":
    unittest.main()
