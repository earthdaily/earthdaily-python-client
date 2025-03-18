import unittest

from earthdaily.earthdatastore.cube_utils import geometry_manager


class TestGeometryManager(unittest.TestCase):
    def setUp(self):
        pass

    def single_geojson(self):
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
        from earthdaily import EarthDataStore

        eds = EarthDataStore()
        eds.datacube(
            "sentinel-2-l2a",
            assets=["blue", "green", "red"],
            datetime="2022-08",
            intersects=gM.to_geopandas(),
        )


if __name__ == "__main__":
    unittest.main()
