import unittest

from earthdaily.datacube import constants
from earthdaily.datacube.models import AggregationMethod


class TestConstants(unittest.TestCase):
    def test_default_dtype(self):
        self.assertEqual(constants.DEFAULT_DTYPE, "float32")
        self.assertIsInstance(constants.DEFAULT_DTYPE, str)

    def test_default_engine(self):
        self.assertEqual(constants.DEFAULT_ENGINE, "odc")
        self.assertIsInstance(constants.DEFAULT_ENGINE, str)

    def test_default_href_path(self):
        self.assertEqual(constants.DEFAULT_HREF_PATH, "alternate.download.href")
        self.assertIsInstance(constants.DEFAULT_HREF_PATH, str)

    def test_default_bbox_crs(self):
        self.assertEqual(constants.DEFAULT_BBOX_CRS, "EPSG:4326")
        self.assertIsInstance(constants.DEFAULT_BBOX_CRS, str)

    def test_default_chunks(self):
        self.assertIsInstance(constants.DEFAULT_CHUNKS, dict)
        self.assertEqual(constants.DEFAULT_CHUNKS["x"], "auto")
        self.assertEqual(constants.DEFAULT_CHUNKS["y"], "auto")
        self.assertEqual(constants.DEFAULT_CHUNKS["time"], 1)

    def test_default_nodata(self):
        self.assertIsNone(constants.DEFAULT_NODATA)

    def test_dim_time(self):
        self.assertEqual(constants.DIM_TIME, "time")
        self.assertIsInstance(constants.DIM_TIME, str)

    def test_dim_x(self):
        self.assertEqual(constants.DIM_X, "x")
        self.assertIsInstance(constants.DIM_X, str)

    def test_dim_y(self):
        self.assertEqual(constants.DIM_Y, "y")
        self.assertIsInstance(constants.DIM_Y, str)

    def test_dim_latitude(self):
        self.assertEqual(constants.DIM_LATITUDE, "latitude")
        self.assertIsInstance(constants.DIM_LATITUDE, str)

    def test_dim_longitude(self):
        self.assertEqual(constants.DIM_LONGITUDE, "longitude")
        self.assertIsInstance(constants.DIM_LONGITUDE, str)

    def test_dim_feature(self):
        self.assertEqual(constants.DIM_FEATURE, "feature")
        self.assertIsInstance(constants.DIM_FEATURE, str)

    def test_dim_zonal_stats(self):
        self.assertEqual(constants.DIM_ZONAL_STATS, "zonal_statistics")
        self.assertIsInstance(constants.DIM_ZONAL_STATS, str)

    def test_dim_bands(self):
        self.assertEqual(constants.DIM_BANDS, "bands")
        self.assertIsInstance(constants.DIM_BANDS, str)

    def test_default_whittaker_beta(self):
        self.assertEqual(constants.DEFAULT_WHITTAKER_BETA, 10000.0)
        self.assertIsInstance(constants.DEFAULT_WHITTAKER_BETA, float)

    def test_default_whittaker_freq(self):
        self.assertEqual(constants.DEFAULT_WHITTAKER_FREQ, "1D")
        self.assertIsInstance(constants.DEFAULT_WHITTAKER_FREQ, str)

    def test_default_temporal_freq(self):
        self.assertEqual(constants.DEFAULT_TEMPORAL_FREQ, "1ME")
        self.assertIsInstance(constants.DEFAULT_TEMPORAL_FREQ, str)

    def test_default_aggregation(self):
        self.assertEqual(constants.DEFAULT_AGGREGATION, "mean")
        self.assertIsInstance(constants.DEFAULT_AGGREGATION, str)
        self.assertIn(constants.DEFAULT_AGGREGATION, AggregationMethod.__args__)

    def test_default_zonal_reducers(self):
        self.assertIsInstance(constants.DEFAULT_ZONAL_REDUCERS, list)
        self.assertEqual(constants.DEFAULT_ZONAL_REDUCERS, ["mean"])
        self.assertIsInstance(constants.DEFAULT_ZONAL_REDUCERS[0], str)

    def test_default_rgb_red(self):
        self.assertEqual(constants.DEFAULT_RGB_RED, "red")
        self.assertIsInstance(constants.DEFAULT_RGB_RED, str)

    def test_default_rgb_green(self):
        self.assertEqual(constants.DEFAULT_RGB_GREEN, "green")
        self.assertIsInstance(constants.DEFAULT_RGB_GREEN, str)

    def test_default_rgb_blue(self):
        self.assertEqual(constants.DEFAULT_RGB_BLUE, "blue")
        self.assertIsInstance(constants.DEFAULT_RGB_BLUE, str)

    def test_default_colormap(self):
        self.assertEqual(constants.DEFAULT_COLORMAP, "Greys")
        self.assertIsInstance(constants.DEFAULT_COLORMAP, str)

    def test_default_col_wrap(self):
        self.assertEqual(constants.DEFAULT_COL_WRAP, 5)
        self.assertIsInstance(constants.DEFAULT_COL_WRAP, int)

    def test_default_time_index(self):
        self.assertEqual(constants.DEFAULT_TIME_INDEX, 0)
        self.assertIsInstance(constants.DEFAULT_TIME_INDEX, int)

    def test_whittaker_alpha(self):
        self.assertEqual(constants.WHITTAKER_ALPHA, 3)
        self.assertIsInstance(constants.WHITTAKER_ALPHA, int)

    def test_whittaker_band_solve(self):
        self.assertEqual(constants.WHITTAKER_BAND_SOLVE, (3, 3))
        self.assertIsInstance(constants.WHITTAKER_BAND_SOLVE, tuple)
        self.assertEqual(len(constants.WHITTAKER_BAND_SOLVE), 2)

    def test_percent_min(self):
        self.assertEqual(constants.PERCENT_MIN, 0)
        self.assertIsInstance(constants.PERCENT_MIN, int)

    def test_percent_max(self):
        self.assertEqual(constants.PERCENT_MAX, 100)
        self.assertIsInstance(constants.PERCENT_MAX, int)


if __name__ == "__main__":
    unittest.main()
