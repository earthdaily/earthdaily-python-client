"""
Test Earth Data Store authentication module.
Credentials must be defined in environment variables or
in the default credentials in order for the test to work.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import toml

from earthdaily.agriculture import EarthDataStore


class TestAuth(unittest.TestCase):
    def setUp(self) -> None:
        """Set up mock credentials and patch external dependencies"""

        # Start patching Auth.read_credentials
        self.patcher_read_credentials = patch("earthdaily.agriculture.earthdatastore.Auth.read_credentials")
        self.mock_read_credentials = self.patcher_read_credentials.start()

        # Set up mocked credentials
        self.mock_credentials = {
            "EDS_AUTH_URL": "https://auth.example.com",
            "EDS_SECRET": "secret_value",
            "EDS_CLIENT_ID": "client_id_value",
        }
        self.mock_read_credentials.return_value = self.mock_credentials

        # Start patching requests.post to prevent real API calls
        self.patcher_requests_post = patch("requests.post")
        self.mock_post = self.patcher_requests_post.start()

        self.mock_post_response = MagicMock()
        self.mock_post_response.json.return_value = {"access_token": "mocked_token"}
        self.mock_post_response.raise_for_status = MagicMock()
        self.mock_post.return_value = self.mock_post_response

        # Start patching pystac_client.Client.open
        self.patcher_client_open = patch("pystac_client.Client.open")
        self.mock_client_open = self.patcher_client_open.start()
        self.mock_client_instance = MagicMock()
        self.mock_client_open.return_value = self.mock_client_instance

        # Create a temporary directory for credentials
        self.temporary_directory = Path(tempfile.mkdtemp())

        # Create JSON credentials file
        self.json_path = self.temporary_directory / "credentials.json"
        with self.json_path.open("w") as f:
            json.dump(self.mock_credentials, f)

        # Create TOML credentials file
        self.toml_path = self.temporary_directory / "credentials.toml"
        with self.toml_path.open("w") as f:
            toml_credentials = {"default": self.mock_credentials, "test_profile": self.mock_credentials}
            toml.dump(toml_credentials, f)

    def tearDown(self) -> None:
        """Stop all patches and clean up temporary files"""
        self.patcher_read_credentials.stop()
        self.patcher_requests_post.stop()
        self.patcher_client_open.stop()
        shutil.rmtree(self.temporary_directory, ignore_errors=True)

    def test_from_json(self) -> None:
        """Ensure EarthDataStore correctly loads credentials from JSON"""
        eds = EarthDataStore(json_path=self.json_path)

        # Ensure the client was opened
        self.assertIsNotNone(eds.client)
        self.mock_client_open.assert_called_once()

    def test_from_input_profile(self) -> None:
        """Ensure EarthDataStore correctly loads credentials from TOML with a profile"""
        eds = EarthDataStore(toml_path=self.toml_path, profile="test_profile")

        # Ensure the client was opened
        self.assertIsNotNone(eds.client)
        self.mock_client_open.assert_called_once()

    def test_from_environment(self) -> None:
        """Ensure EarthDataStore correctly loads credentials from environment variables"""
        os.environ["EDS_AUTH_URL"] = self.mock_credentials["EDS_AUTH_URL"]
        os.environ["EDS_SECRET"] = self.mock_credentials["EDS_SECRET"]
        os.environ["EDS_CLIENT_ID"] = self.mock_credentials["EDS_CLIENT_ID"]

        eds = EarthDataStore()

        # Ensure the client was opened
        self.assertIsNotNone(eds.client)
        self.mock_client_open.assert_called_once()


if __name__ == "__main__":
    unittest.main()
