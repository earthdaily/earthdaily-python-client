import unittest

from earthdaily.agriculture import EarthDataStore
from earthdaily.agriculture.datasets import load_pivot, load_pivot_corumba


class TestEarthDataStore(unittest.TestCase):
    def setUp(self):
        self.eds = EarthDataStore()
        self.pivot = load_pivot()
        self.pivot_corumba = load_pivot_corumba()

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
            mask_with=["native"],
        )
        self.assertEqual(list(datacube.data_vars.keys()), ["blue", "green", "red"])


if __name__ == "__main__":
    unittest.main()
