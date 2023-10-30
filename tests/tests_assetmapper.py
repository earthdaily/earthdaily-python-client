import unittest

from earthdaily.earthdatastore.cube_utils.asset_mapper import AssetMapper

class TestZonalStats(unittest.TestCase):
    def test_unknow_collection(self):
        collection = "earthdaily-unknow-collection"
        assets = ['blue','green','red','lambda']
        assert(AssetMapper().map_collection_bands(collection,assets) == assets)

    def test_return_same_dict(self):
        collection = "sentinel-2-l2a"
        assets = {"key":"value","source":"target","sensorasset":"myoutputband"}
        assert(AssetMapper().map_collection_bands(collection,assets) == assets)


    def test_sentinel2(self):
        collection = "sentinel-2-l2a"
        assets = ['blue','green','red','rededge74', "missing_band"]
        assets_s2 = ['blue','green','red','rededge2']
        assert list(AssetMapper().map_collection_bands(collection,assets).keys()) == assets_s2

    def test_venus_rededge(self):
        collection = "venus-l2a"
        rededges = {"rededge70": "image_file_SRE_B08",
                    "rededge74": "image_file_SRE_B09",
                    "rededge78": "image_file_SRE_B10"}
    
        assert list(AssetMapper().map_collection_bands(collection,rededges.keys()).keys()) == list(rededges.values())


if __name__ == "__main__":
    unittest.main()
