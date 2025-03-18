import unittest

from earthdaily.earthdatastore.cube_utils.asset_mapper import AssetMapper


class TestAssetMapper(unittest.TestCase):
    def setUp(self):
        self.aM = AssetMapper()

    def test_unknow_collection(self):
        collection = "earthdaily-unknow-collection"
        assets = ["blue", "green", "red", "lambda"]
        self.assertEqual(self.aM.map_collection_assets(collection, assets), assets)
        with self.assertRaises(NotImplementedError):
            self.aM._collection_exists(collection, raise_warning=True)

    def test_return_same_dict(self):
        collection = "sentinel-2-l2a"
        assets = {"key": "value", "source": "target", "sensorasset": "myoutputband"}
        self.assertEqual(self.aM.map_collection_assets(collection, assets), assets)

    def test_sentinel2(self):
        collection = "sentinel-2-l2a"
        assets = ["blue", "green", "red", "rededge74", "missing_band"]
        assets_s2 = ["blue", "green", "red", "rededge2"]
        self.assertEqual(
            list(self.aM.map_collection_assets(collection, assets).keys()), assets_s2
        )

    def test_venus_rededge(self):
        collection = "venus-l2a"
        rededges = {
            "rededge70": "image_file_SRE_B08",
            "rededge74": "image_file_SRE_B09",
            "rededge78": "image_file_SRE_B10",
        }

        self.assertEqual(
            list(self.aM.map_collection_assets(collection, rededges.keys()).keys()),
            list(rededges.values()),
        )


if __name__ == "__main__":
    unittest.main()
