import unittest

import earthdaily


class TestEarthDataStore(unittest.TestCase):
    def setUp(self):
        self.eds = earthdaily.earthdatastore.Auth()
        self.pivot = earthdaily.datasets.load_pivot()

    def test_rescale_on_venus(self):
        collection = "venus-l2a"
        theia_location = "MEAD"
        max_cloud_cover = 20

        query = {
            "theia:location": {"eq": theia_location},
            "eo:cloud_cover": {"lt": max_cloud_cover},
        }

        items = self.eds.search(collection, query=query, max_items=1)
        crs = items[0].properties["proj:epsg"]
        gsd = items[0].properties["gsd"]

        bands_info = (
            items[0].assets["image_file_SRE_B3"].extra_fields["raster:bands"][0]
        )
        scale, offset = bands_info["scale"], bands_info["offset"]
        for rescale in True, False:
            datacube = self.eds.datacube(
                collection,
                assets=["image_file_SRE_B3"],
                rescale=rescale,
                search_kwargs=dict(query=query, max_items=1),
                resolution=gsd,
                crs=crs,
            )

            # self.assertEqual(datacube.rio.width,9374)
            # self.assertEqual(datacube.rio.height,10161)
            self.assertEqual(datacube.time.size, 1)
            blue = (
                datacube["image_file_SRE_B3"]
                .isel(x=4000, y=4000, time=0)
                .data.compute()
            )
            if rescale is False:
                blue = blue * scale + offset
            self.assertAlmostEqual(blue, 0.136)

    def test_sentinel1(self):
        collection = "sentinel-1-rtc"
        datacube = self.eds.datacube(
            collection,
            assets=["vh", "vv"],
            intersects=self.pivot,
            datetime="2022-01",
        )
        self.assertEqual(list(datacube.data_vars.keys()), ["vh", "vv"])

    def test_sentinel2(self):
        collection = "sentinel-2-l2a"
        datacube = self.eds.datacube(
            collection,
            assets=["blue", "green", "red"],
            intersects=self.pivot,
            datetime="2023-07-01",
        )
        self.assertEqual(list(datacube.data_vars.keys()), ["blue", "green", "red"])


if __name__ == "__main__":
    unittest.main()
