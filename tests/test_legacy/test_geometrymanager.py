import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pystac import ItemCollection

from earthdaily import EDSClient, EDSConfig
from earthdaily.legacy.earthdatastore.cube_utils import geometry_manager


class TestGeometryManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Store original environment
        cls.original_env = dict(os.environ)

    def setUp(self):
        os.environ.clear()
        os.environ.update(self.original_env)
        os.environ["EDS_CLIENT_ID"] = "env_client_id"
        os.environ["EDS_SECRET"] = "env_client_secret"
        os.environ["EDS_AUTH_URL"] = "https://env_token_url.com"
        os.environ["EDS_API_URL"] = "https://EDS_API_URL.com"

        self.patcher_requests_post = patch("earthdaily._auth_client.requests.Session.post")
        self.mock_post = self.patcher_requests_post.start()

        self.mock_post_response = MagicMock()
        self.mock_post_response.json.return_value = {"access_token": "mock_token", "expires_in": 3600}
        self.mock_post_response.raise_for_status = MagicMock()
        self.mock_post.return_value = self.mock_post_response

        TEST_DATA_PATH = Path(__file__).parent / "test_data.json"

        with TEST_DATA_PATH.open("r") as f:
            DUMPED_JSON_RESPONSE = json.load(f)
        """Mock pystac_client.Client.open and .search to return a real pystac ItemCollection"""
        self.patcher_client_open = patch("pystac_client.Client.open")
        self.mock_client_open = self.patcher_client_open.start()

        # Mock Client instance
        self.mock_client_instance = MagicMock()
        self.mock_client_open.return_value = self.mock_client_instance

        # Convert dumped JSON response into an actual pystac ItemCollection object
        self.mock_item_collection = ItemCollection.from_dict(DUMPED_JSON_RESPONSE)

        # Mock the .search() method to return a fake search result that calls .item_collection()
        self.mock_search_results = MagicMock()
        self.mock_search_results.item_collection.return_value = self.mock_item_collection
        self.mock_client_instance.search.return_value = self.mock_search_results

    def tearDown(self):
        """Stop all patches after each test"""
        self.patcher_client_open.stop()
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_single_geojson(self):
        """Ensure GeometryManager correctly initializes with a single geometry"""
        geom = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"type": "forest"},
                    "geometry": {
                        "coordinates": [
                            [
                                [1.248715854758899, 43.66258153536606],
                                [1.248715854758899, 43.661751304559004],
                                [1.2499517768647195, 43.661751304559004],
                                [1.2499517768647195, 43.66258153536606],
                                [1.248715854758899, 43.66258153536606],
                            ]
                        ],
                        "type": "Polygon",
                    },
                }
            ],
        }
        geometry_manager.GeometryManager(geom)

    def test_two_geometry_geojson(self):
        """Ensure EarthDataStore correctly processes multiple geometries in datacube"""
        geom = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"type": "crop"},
                    "geometry": {
                        "coordinates": [
                            [
                                [1.2527767416787583, 43.67384173712989],
                                [1.2527767416787583, 43.67184102384948],
                                [1.255895973661012, 43.67184102384948],
                                [1.255895973661012, 43.67384173712989],
                                [1.2527767416787583, 43.67384173712989],
                            ]
                        ],
                        "type": "Polygon",
                    },
                    "id": 1,
                },
                {
                    "type": "Feature",
                    "properties": {"type": "forest"},
                    "geometry": {
                        "coordinates": [
                            [
                                [1.248715854758899, 43.66258153536606],
                                [1.248715854758899, 43.661751304559004],
                                [1.2499517768647195, 43.661751304559004],
                                [1.2499517768647195, 43.66258153536606],
                                [1.248715854758899, 43.66258153536606],
                            ]
                        ],
                        "type": "Polygon",
                    },
                    "id": 2,
                },
            ],
        }

        gM = geometry_manager.GeometryManager(geom)
        eds = EDSClient(EDSConfig())

        # Call datacube, which internally calls search().item_collection()
        items = eds.legacy.datacube(
            "sentinel-2-l2a", assets=["blue", "green", "red"], datetime="2022-08", intersects=gM.to_geopandas()
        )
        self.mock_client_instance.search.assert_called_once()
        self.mock_search_results.item_collection.assert_called_once()
        self.assertEqual(items, self.mock_item_collection)

        expected_dims = ("time", "y", "x")
        self.assertEqual(set(items.sizes.keys()), set(expected_dims))
        self.assertEqual(items.sizes["time"], 6)
        self.assertEqual(items.sizes["y"], 134)
        self.assertEqual(items.sizes["x"], 60)


if __name__ == "__main__":
    unittest.main()
