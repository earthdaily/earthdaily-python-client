import unittest

import geopandas as gpd
import numpy as np
import xarray as xr
from pyproj import CRS
from earthdaily.earthdatastore import cube_utils

# load the geoJson file
gdf = gpd.read_file("geometries.geojson")

# set EPSG 4326
gdf = gdf.to_crs(CRS.from_epsg(4326))


dataset = xr.open_zarr('./zonal_stat_test_datacube.zarr')
dataset.load()


class TestZonalStats(unittest.TestCase):
    def test_max_data(self):
        # Test case to valid 'max' data result
        result = cube_utils.zonal_stats(dataset, gdf, operations=["max"], all_touched=False)
        # Add assertions to check if the result is as expected
        expected_result = np.array([[[7.]], [[4.]]])

        assert result['data'] is not None
        assert result['data'].values is not None and isinstance(result['data'].values, np.ndarray)
        assert np.allclose(result['data'].values, expected_result, atol=1e-6)

    def test_min_data(self):
        # Test case to valid 'min' data result
        result = cube_utils.zonal_stats(dataset, gdf, operations=["min"], all_touched=False)
        # Add assertions to check if the result is as expected
        expected_result = np.array([[[3.]], [[3.]]])

        assert result['data'] is not None
        assert result['data'].values is not None and isinstance(result['data'].values, np.ndarray)
        assert np.allclose(result['data'].values, expected_result, atol=1e-6)

    def test_count_data(self):
        # Test case to valid 'count' data result
        result = cube_utils.zonal_stats(dataset, gdf, operations=["count"], all_touched=False)
        # Add assertions to check if the result is as expected
        expected_result = np.array([[[38.]], [[11.]]])

        assert result['data'] is not None
        assert result['data'].values is not None and isinstance(result['data'].values, np.ndarray)
        assert np.allclose(result['data'].values, expected_result, atol=1e-6)

    def test_mean_data(self):
        # Test case to valid 'mean' data result
        result = cube_utils.zonal_stats(dataset, gdf, operations=["mean"], all_touched=False)
        # Add assertions to check if the result is as expected
        expected_result = np.array([[[4.02631579]], [[3.90909091]]])

        assert result['data'] is not None
        assert result['data'].values is not None and isinstance(result['data'].values, np.ndarray)
        assert np.allclose(result['data'].values, expected_result, atol=1e-6)

    def test_median_data(self):
        # Test case to valid 'median' data result
        result = cube_utils.zonal_stats(dataset, gdf, operations=["median"], all_touched=False)
        # Add assertions to check if the result is as expected
        expected_result = np.array([[[4.0]], [[4.0]]])

        assert result['data'] is not None
        assert result['data'].values is not None and isinstance(result['data'].values, np.ndarray)
        assert np.allclose(result['data'].values, expected_result, atol=1e-6)

    def test_std_data(self):
        # Test case to valid 'std' data result
        result = cube_utils.zonal_stats(dataset, gdf, operations=["std"], all_touched=False)
        # Add assertions to check if the result is as expected
        expected_result = np.array([[[0.77754141]], [[0.28747979]]])

        assert result['data'] is not None
        assert result['data'].values is not None and isinstance(result['data'].values, np.ndarray)
        assert np.allclose(result['data'].values, expected_result, atol=1e-6)

    def test_mode_data(self):
        # Test case to valid 'mode' data result
        result = cube_utils.zonal_stats(dataset, gdf, operations=["mode"], all_touched=False)
        # Add assertions to check if the result is as expected
        expected_result = np.array([[[4.0]], [[4.0]]])

        assert result['data'] is not None
        assert result['data'].values is not None and isinstance(result['data'].values, np.ndarray)
        assert np.allclose(result['data'].values, expected_result, atol=1e-6)

    def test_all_data(self):
        # Test case to valid 'all' data result and ignore 'wrong_operation'
        result = cube_utils.zonal_stats(dataset, gdf,
                                        operations=["count", "max", "mean", "median", "min", "mode", "std"],
                                        all_touched=False)
        # Add assertions to check if the result is as expected
        expected_result = np.array([[[38.], [7.], [4.02631579], [4.], [3.], [4.], [0.77754141]],
                                    [[11.], [4.], [3.90909091], [4.], [3.], [4.], [0.28747979]]])

        assert result['data'] is not None
        assert result['data'].values is not None and isinstance(result['data'].values, np.ndarray)
        assert np.allclose(result['data'].values, expected_result, atol=1e-6)

    def test_invalid_operation(self):
        # Test case for invalid input data
        with self.assertRaises(ValueError):
            cube_utils.zonal_stats(dataset, gdf, operations=["wrong_operation"], all_touched=False)
