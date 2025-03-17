"""
Test Earth Data Store authentification module.
Credentials must be defined in environment variables or
in the default credentials in order for the test to work.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import toml

from earthdaily import EarthDataStore
from earthdaily.earthdatastore import Auth


class TestAuth(unittest.TestCase):
    def setUp(self) -> None:
        self.credentials = Auth.read_credentials()
        self.temporary_directory = Path(tempfile.mkdtemp())

        # Create JSON credentials
        self.json_path = self.temporary_directory / "credentials.json"
        with self.json_path.open("w") as f:
            json.dump(self.credentials, f)

        # Create TOML credentials
        self.toml_path = self.temporary_directory / "credentials.toml"
        with self.toml_path.open("w") as f:
            toml_credentials = {
                "default": self.credentials,
                "test_profile": self.credentials,
            }
            toml.dump(toml_credentials, f)

    def tearDown(self) -> None:
        self.json_path.unlink()
        self.toml_path.unlink()
        self.temporary_directory.rmdir()

    def test_from_json(self) -> None:
        EarthDataStore(json_path=self.json_path)

    def test_from_input_profile(self) -> None:
        EarthDataStore(toml_path=self.toml_path, profile="test_profile")

    def test_from_environment(self) -> None:
        # Ensure environment variables are set
        os.environ["EDS_AUTH_URL"] = self.credentials["EDS_AUTH_URL"]
        os.environ["EDS_SECRET"] = self.credentials["EDS_SECRET"]
        os.environ["EDS_CLIENT_ID"] = self.credentials["EDS_CLIENT_ID"]
        EarthDataStore()


if __name__ == "__main__":
    unittest.main()
