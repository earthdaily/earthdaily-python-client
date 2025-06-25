import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import xarray as xr
from pystac import ItemCollection

from earthdaily import EDSClient, EDSConfig
from earthdaily.legacy.datasets import load_pivot, load_pivot_corumba


class TestEarthDataStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Store original environment
        cls.original_env = dict(os.environ)

    def setUp(self):
        """Mock pystac_client.Client.open and .search() to return mock ItemCollection"""
        os.environ.clear()
        os.environ.update(self.original_env)
        os.environ["EDS_CLIENT_ID"] = "env_client_id"
        os.environ["EDS_SECRET"] = "env_client_secret"
        os.environ["EDS_AUTH_URL"] = "env_token_url"
        os.environ["EDS_API_URL"] = "https://EDS_API_URL.com"

        self.patcher_requests_post = patch("earthdaily._auth_client.requests.Session.post")
        self.mock_post = self.patcher_requests_post.start()

        self.mock_post_response = MagicMock()
        self.mock_post_response.json.return_value = {"access_token": "mock_token", "expires_in": 3600}
        self.mock_post_response.raise_for_status = MagicMock()
        self.mock_post.return_value = self.mock_post_response
        self.patcher_client_open = patch("pystac_client.Client.open")
        self.mock_client_open = self.patcher_client_open.start()

        self.mock_client_instance = MagicMock()
        self.mock_client_open.return_value = self.mock_client_instance
        s2_test_data_path = Path(__file__).parent / "test_data_s2_mask.json"
        with s2_test_data_path.open("r") as f:
            s2_json_response = json.load(f)
        self.mock_item_collection_s2 = ItemCollection.from_dict(s2_json_response)

        s1_test_data_path = Path(__file__).parent / "test_data_s1.json"
        with s1_test_data_path.open("r") as f:
            s1_json_response = json.load(f)
        self.mock_item_collection_s1 = ItemCollection.from_dict(s1_json_response)

        self.mock_search_results_s1 = MagicMock()
        self.mock_search_results_s1.item_collection.return_value = self.mock_item_collection_s1
        self.mock_search_results_s2 = MagicMock()
        self.mock_search_results_s2.item_collection.return_value = self.mock_item_collection_s2

        def mock_search(collections, *args, **kwargs):
            if "sentinel-1-rtc" in collections:
                return self.mock_search_results_s1
            elif "sentinel-2-l2a" in collections:
                return self.mock_search_results_s2
            return MagicMock()

        self.mock_client_instance.search.side_effect = mock_search

        self.eds = EDSClient(EDSConfig())
        self.pivot = load_pivot()
        self.pivot_corumba = load_pivot_corumba()

    def tearDown(self):
        """Stop all patches after each test"""
        self.patcher_client_open.stop()

    def test_sentinel1(self):
        """Ensure Sentinel-1 RTC datacube correctly loads mocked data"""
        collection = "sentinel-1-rtc"
        datacube = self.eds.legacy.datacube(
            collection,
            assets=["vh", "vv"],
            intersects=self.pivot,
            datetime="2022-01",
        )
        self.mock_client_instance.search.assert_called()
        self.mock_search_results_s1.item_collection.assert_called_once()
        self.assertIsInstance(datacube, xr.Dataset)
        self.assertEqual(set(datacube.data_vars.keys()), {"vh", "vv"})
        self.assertIn("time", datacube.coords)

    def test_sentinel2(self):
        """Ensure Sentinel-2 datacube correctly loads mocked data"""
        collection = "sentinel-2-l2a"
        datacube = self.eds.legacy.datacube(
            collection,
            assets=["blue", "green", "red"],
            intersects=self.pivot,
            datetime="2023-07-01",
            mask_with=["native"],
        )

        self.mock_client_instance.search.assert_called()
        self.mock_search_results_s2.item_collection.assert_called_once()
        self.assertIsInstance(datacube, xr.Dataset)
        self.assertEqual(set(datacube.data_vars.keys()), {"blue", "green", "red"})
        self.assertIn("time", datacube.coords)


if __name__ == "__main__":
    unittest.main()
