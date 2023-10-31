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
        
        items  = self.eds.search(collection, query=query, max_items=1)
        crs = items[0].properties['proj:epsg']
        gsd = items[0].properties['gsd']


        datacube = self.eds.datacube(collection, assets=['image_file_SRE_B3'], search_kwargs=dict(query=query, max_items=1),resolution=gsd,crs=crs)
    
        self.assertEqual(datacube.rio.width,9374)
        self.assertEqual(datacube.rio.height,10161)
        self.assertEqual(datacube.time.size,1)
        blue = datacube['image_file_SRE_B3'].isel(x=5000,y=5000,time=0).data.compute()
        self.assertEqual(blue,0.028999999999999998)
        
    
    def test_sentinel1(self):
        # TODO : implement s1
        collection = "sentinel-1-rtc"
        
        # datacube = self.eds.datacube(collection, bbox=bbox)

        
if __name__ == "__main__":
    unittest.main()
