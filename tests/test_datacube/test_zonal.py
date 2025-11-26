import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from earthdaily.datacube._zonal import (
    _compute_stats,
    _compute_zonal_statistics,
    _format_output,
    _rasterize_geometries,
    compute_zonal_stats,
)
from earthdaily.datacube.constants import DIM_FEATURE, DIM_ZONAL_STATS


class TestComputeZonalStats(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=3, freq="D")
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(3, 20, 20) * 100),
                "band2": (["time", "y", "x"], np.random.rand(3, 20, 20) * 100),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.dataset.rio.write_crs("EPSG:4326", inplace=True)
        self.dataset.rio.write_transform(inplace=True)

        self.geometry = gpd.GeoDataFrame(geometry=[box(2, 2, 8, 8)], crs="EPSG:4326")

    def test_compute_zonal_stats_with_defaults(self):
        result = compute_zonal_stats(self.dataset, self.geometry)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn(DIM_FEATURE, result.dims)
        self.assertIn(DIM_ZONAL_STATS, result.dims)

    def test_compute_zonal_stats_with_geodataframe(self):
        result = compute_zonal_stats(self.dataset, self.geometry)
        self.assertIsInstance(result, xr.Dataset)

    def test_compute_zonal_stats_with_dict_geometry(self):
        geometry_dict = {
            "type": "Polygon",
            "coordinates": [[[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]],
        }
        result = compute_zonal_stats(self.dataset, geometry_dict)
        self.assertIsInstance(result, xr.Dataset)

    def test_compute_zonal_stats_with_string_geometry(self):
        geometry_str = '{"type": "Polygon", "coordinates": [[[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]]}'
        result = compute_zonal_stats(self.dataset, geometry_str)
        self.assertIsInstance(result, xr.Dataset)

    def test_compute_zonal_stats_with_multiple_reducers(self):
        reducers = ["mean", "median", "std", "min", "max"]
        result = compute_zonal_stats(self.dataset, self.geometry, reducers=reducers)
        self.assertIn(DIM_ZONAL_STATS, result.coords)
        self.assertEqual(len(result[DIM_ZONAL_STATS]), len(reducers))

    def test_compute_zonal_stats_with_all_touched_false(self):
        result = compute_zonal_stats(self.dataset, self.geometry, all_touched=False)
        self.assertIsInstance(result, xr.Dataset)

    def test_compute_zonal_stats_with_preserve_columns_false(self):
        geometry_with_attrs = gpd.GeoDataFrame(
            {"name": ["test"], "value": [42]}, geometry=[box(2, 2, 8, 8)], crs="EPSG:4326"
        )
        result = compute_zonal_stats(self.dataset, geometry_with_attrs, preserve_columns=False)
        self.assertIsInstance(result, xr.Dataset)
        self.assertNotIn("name", result.coords)
        self.assertNotIn("value", result.coords)

    def test_compute_zonal_stats_with_preserve_columns_true(self):
        geometry_with_attrs = gpd.GeoDataFrame(
            {"name": ["test"], "value": [42]}, geometry=[box(2, 2, 8, 8)], crs="EPSG:4326"
        )
        result = compute_zonal_stats(self.dataset, geometry_with_attrs, preserve_columns=True)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("name", result.coords)
        self.assertIn("value", result.coords)

    def test_compute_zonal_stats_with_lazy_load_false(self):
        result = compute_zonal_stats(self.dataset, self.geometry, lazy_load=False)
        self.assertIsInstance(result, xr.Dataset)

    def test_compute_zonal_stats_with_lazy_load_true(self):
        result = compute_zonal_stats(self.dataset, self.geometry, lazy_load=True)
        self.assertIsInstance(result, xr.Dataset)

    def test_compute_zonal_stats_multiple_geometries(self):
        geometries = gpd.GeoDataFrame(geometry=[box(2, 2, 5, 5), box(6, 6, 9, 9)], crs="EPSG:4326")
        result = compute_zonal_stats(self.dataset, geometries)
        self.assertGreaterEqual(len(result[DIM_FEATURE]), 1)

    def test_compute_zonal_stats_preserved_columns_align_with_intersections(self):
        geometries = gpd.GeoDataFrame(
            {
                "name": ["outside", "inside"],
                "value": [0, 99],
                "geometry": [box(100, 100, 110, 110), box(2, 2, 8, 8)],
            },
            crs="EPSG:4326",
        )
        result = compute_zonal_stats(self.dataset, geometries, preserve_columns=True)
        self.assertEqual(result.coords["name"].item(), "inside")
        self.assertEqual(result.coords["value"].item(), 99)

    def test_compute_zonal_stats_no_intersection_raises_error(self):
        from rioxarray.exceptions import NoDataInBounds

        geometry_no_intersect = gpd.GeoDataFrame(geometry=[box(100, 100, 110, 110)], crs="EPSG:4326")
        with self.assertRaises((ValueError, NoDataInBounds)):
            compute_zonal_stats(self.dataset, geometry_no_intersect)


class TestComputeZonalStatistics(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=3, freq="D")
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(3, 20, 20) * 100),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.dataset.rio.write_crs("EPSG:4326", inplace=True)
        self.dataset.rio.write_transform(inplace=True)

        self.geometries = gpd.GeoDataFrame(geometry=[box(2, 2, 8, 8)], crs="EPSG:4326")

    def test_compute_zonal_statistics_basic(self):
        result = _compute_zonal_statistics(self.dataset, self.geometries)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn(DIM_FEATURE, result.dims)

    def test_compute_zonal_statistics_with_custom_reducers(self):
        reducers = ["mean", "std"]
        result = _compute_zonal_statistics(self.dataset, self.geometries, reducers=reducers)
        self.assertIn(DIM_ZONAL_STATS, result.coords)
        self.assertEqual(len(result[DIM_ZONAL_STATS]), len(reducers))


class TestRasterizeGeometries(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)

        self.dataset = xr.Dataset(
            {
                "band1": (["y", "x"], np.random.rand(20, 20)),
            },
            coords={"y": y, "x": x},
        )
        self.dataset.rio.write_crs("EPSG:4326", inplace=True)
        self.dataset.rio.write_transform(inplace=True)

        self.geometries = gpd.GeoDataFrame(geometry=[box(2, 2, 8, 8)], crs="EPSG:4326")

    def test_rasterize_geometries_basic(self):
        features, positions = _rasterize_geometries(self.geometries, self.dataset, all_touched=True)
        self.assertIsInstance(features, list)
        self.assertIsInstance(positions, list)
        self.assertEqual(len(features), len(self.geometries))
        self.assertEqual(len(positions), len(self.geometries))

    def test_rasterize_geometries_all_touched_false(self):
        features, positions = _rasterize_geometries(self.geometries, self.dataset, all_touched=False)
        self.assertIsInstance(features, list)
        self.assertIsInstance(positions, list)

    def test_rasterize_geometries_multiple_geometries(self):
        geometries = gpd.GeoDataFrame(geometry=[box(2, 2, 5, 5), box(6, 6, 9, 9)], crs="EPSG:4326")
        features, positions = _rasterize_geometries(geometries, self.dataset, all_touched=True)
        self.assertEqual(len(features), len(geometries))
        self.assertEqual(len(positions), len(geometries))

    def test_rasterize_geometries_no_intersection(self):
        geometries = gpd.GeoDataFrame(geometry=[box(100, 100, 110, 110)], crs="EPSG:4326")
        features, positions = _rasterize_geometries(geometries, self.dataset, all_touched=True)
        self.assertEqual(len(features), 0)
        self.assertEqual(len(positions), len(geometries))


class TestComputeStats(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=3, freq="D")
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(3, 20, 20) * 100),
            },
            coords={"time": times, "y": y, "x": x},
        )

        y_coords = np.array([5, 6, 7, 8, 9])
        x_coords = np.array([5, 6, 7, 8, 9])
        self.positions = [(y_coords, x_coords)]

    def test_compute_stats_with_mean(self):
        reducers = ["mean"]
        result = _compute_stats(self.dataset, self.positions, reducers)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_compute_stats_with_multiple_reducers(self):
        reducers = ["mean", "median", "std", "min", "max"]
        result = _compute_stats(self.dataset, self.positions, reducers)
        self.assertIn(DIM_ZONAL_STATS, result.coords)
        self.assertEqual(len(result[DIM_ZONAL_STATS]), len(reducers))

    def test_compute_stats_with_mode(self):
        reducers = ["mode"]
        result = _compute_stats(self.dataset, self.positions, reducers)
        self.assertIsInstance(result, xr.Dataset)

    def test_compute_stats_with_sum(self):
        reducers = ["sum"]
        result = _compute_stats(self.dataset, self.positions, reducers)
        self.assertIsInstance(result, xr.Dataset)

    def test_compute_stats_with_empty_positions(self):
        reducers = ["mean"]
        empty_positions = [()]
        result = _compute_stats(self.dataset, empty_positions, reducers)
        self.assertIsInstance(result, xr.Dataset)

    def test_compute_stats_multiple_positions(self):
        y_coords1 = np.array([5, 6])
        x_coords1 = np.array([5, 6])
        y_coords2 = np.array([7, 8])
        x_coords2 = np.array([7, 8])
        positions = [(y_coords1, x_coords1), (y_coords2, x_coords2)]
        reducers = ["mean"]
        result = _compute_stats(self.dataset, positions, reducers)
        self.assertEqual(len(result[DIM_FEATURE]), len(positions))


class TestFormatOutput(unittest.TestCase):
    def setUp(self):
        self.stats = xr.Dataset(
            {
                "band1": (
                    [DIM_FEATURE, DIM_ZONAL_STATS],
                    np.random.rand(2, 3),
                ),
            },
            coords={DIM_FEATURE: [0, 1], DIM_ZONAL_STATS: ["mean", "median", "std"]},
        )

        self.features = [
            {"geometry": box(0, 0, 1, 1), "index": 0},
            {"geometry": box(2, 2, 3, 3), "index": 1},
        ]

        self.geometries = gpd.GeoDataFrame(
            {"name": ["A", "B"], "value": [10, 20]},
            geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
            crs="EPSG:4326",
        )

    def test_format_output_basic(self):
        result = _format_output(self.stats, self.features, self.geometries, preserve_columns=False)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("geometry", result.coords)

    def test_format_output_with_preserve_columns_true(self):
        result = _format_output(self.stats, self.features, self.geometries, preserve_columns=True)
        self.assertIn("geometry", result.coords)
        self.assertIn("name", result.coords)
        self.assertIn("value", result.coords)

    def test_format_output_with_preserve_columns_false(self):
        result = _format_output(self.stats, self.features, self.geometries, preserve_columns=False)
        self.assertIn("geometry", result.coords)
        self.assertNotIn("name", result.coords)
        self.assertNotIn("value", result.coords)

    def test_format_output_geometry_wkt(self):
        result = _format_output(self.stats, self.features, self.geometries, preserve_columns=False)
        geometry_coords = result.coords["geometry"].values
        self.assertEqual(len(geometry_coords), len(self.features))
        self.assertTrue(all(isinstance(g, str) for g in geometry_coords))

    def test_format_output_respects_feature_indices(self):
        reordered_features = [{"geometry": box(2, 2, 3, 3), "index": 1}]
        result = _format_output(
            self.stats.isel({DIM_FEATURE: slice(0, 1)}),
            reordered_features,
            self.geometries,
            preserve_columns=True,
        )
        self.assertEqual(result.coords["name"].item(), "B")
        self.assertEqual(result.coords["value"].item(), 20)


if __name__ == "__main__":
    unittest.main()
