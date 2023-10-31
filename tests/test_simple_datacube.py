import unittest

import earthdaily


class TestEarthDataStore(unittest.TestCase):
    def setUp(self):
        self.eds = earthdaily.earthdatastore.Auth()

    def test_venus(self):
        collection = "venus-l2a"
        theia_location = "MEAD"
        max_cloud_cover = 20

        query = {
            "theia:location": {"eq": theia_location},
            "eo:cloud_cover": {"lt": max_cloud_cover},
        }
        
        items  = self.eds.search(collection, query=query, max_items=2)


        datacube = self.eds.datacube(collection, search_kwargs=dict(query=query, max_items=2),resolution=5, crs=3857, rescale=True)
        self.assertEqual(datacube.rio.width,12054)
        self.assertEqual(datacube.rio.height,13178)
        self.assertEqual(datacube.time.size,2)
        blue = datacube['image_file_SRE_B3'].isel(x=5000,y=5000,time=0).data.compute()
        self.assertEqual(blue,0.134)
        
if __name__ == "__main__":
    unittest.main()
