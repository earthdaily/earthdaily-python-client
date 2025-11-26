import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rioxarray.exceptions import NoDataInBounds
from shapely.geometry import box

from earthdaily.datacube._operations import (
    clip_datacube,
    merge_datacubes,
    rechunk_datacube,
    select_time_range,
)
from earthdaily.datacube.exceptions import DatacubeMergeError, DatacubeOperationError


class TestClipDatacube(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=3, freq="D")
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(3, 20, 20)),
                "band2": (["time", "y", "x"], np.random.rand(3, 20, 20)),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.dataset.rio.write_crs("EPSG:4326", inplace=True)
        self.dataset.rio.write_transform(inplace=True)

    def test_clip_with_geodataframe(self):
        geometry = gpd.GeoDataFrame(geometry=[box(2, 2, 8, 8)], crs="EPSG:4326")
        result = clip_datacube(self.dataset, geometry)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)
        self.assertIn("band2", result.data_vars)

    def test_clip_with_dict_geometry(self):
        geometry_dict = {
            "type": "Polygon",
            "coordinates": [[[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]],
        }
        result = clip_datacube(self.dataset, geometry_dict)
        self.assertIsInstance(result, xr.Dataset)

    def test_clip_with_string_geometry(self):
        geometry_str = '{"type": "Polygon", "coordinates": [[[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]]}'
        result = clip_datacube(self.dataset, geometry_str)
        self.assertIsInstance(result, xr.Dataset)

    @patch("earthdaily.datacube._operations.geometry_to_geopandas")
    def test_clip_handles_no_data_in_bounds(self, mock_geometry_to_gdf):
        mock_gdf = MagicMock()
        mock_gdf.to_crs.return_value = mock_gdf
        mock_gdf.total_bounds = [2, 2, 8, 8]
        mock_gdf.geometry = MagicMock()
        mock_geometry_to_gdf.return_value = mock_gdf

        self.dataset.rio.clip_box = MagicMock(side_effect=NoDataInBounds("No data"))

        with self.assertRaises(DatacubeOperationError) as context:
            clip_datacube(self.dataset, MagicMock())

        self.assertIn("No data found in the specified geometry bounds", str(context.exception))

    def test_clip_preserves_bands(self):
        geometry = gpd.GeoDataFrame(geometry=[box(2, 2, 8, 8)], crs="EPSG:4326")
        result = clip_datacube(self.dataset, geometry)
        self.assertIn("band1", result.data_vars)
        self.assertIn("band2", result.data_vars)


class TestMergeDatacubes(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=3, freq="D")
        x = np.arange(10)
        y = np.arange(10)

        self.dataset1 = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(3, 10, 10)),
                "band2": (["time", "y", "x"], np.random.rand(3, 10, 10)),
            },
            coords={"time": times, "y": y, "x": x},
        )

        self.dataset2 = xr.Dataset(
            {
                "band3": (["time", "y", "x"], np.random.rand(3, 10, 10)),
                "band4": (["time", "y", "x"], np.random.rand(3, 10, 10)),
            },
            coords={"time": times, "y": y, "x": x},
        )

    def test_merge_with_default_compat(self):
        result = merge_datacubes(self.dataset1, self.dataset2)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)
        self.assertIn("band2", result.data_vars)
        self.assertIn("band3", result.data_vars)
        self.assertIn("band4", result.data_vars)

    def test_merge_with_override_compat(self):
        result = merge_datacubes(self.dataset1, self.dataset2, compat="override")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)
        self.assertIn("band3", result.data_vars)

    def test_merge_with_identical_compat(self):
        result = merge_datacubes(self.dataset1, self.dataset2, compat="identical")
        self.assertIsInstance(result, xr.Dataset)

    def test_merge_with_equals_compat(self):
        result = merge_datacubes(self.dataset1, self.dataset2, compat="equals")
        self.assertIsInstance(result, xr.Dataset)

    def test_merge_with_broadcast_equals_compat(self):
        result = merge_datacubes(self.dataset1, self.dataset2, compat="broadcast_equals")
        self.assertIsInstance(result, xr.Dataset)

    def test_merge_with_no_conflicts_compat(self):
        result = merge_datacubes(self.dataset1, self.dataset2, compat="no_conflicts")
        self.assertIsInstance(result, xr.Dataset)

    def test_merge_with_minimal_compat(self):
        result = merge_datacubes(self.dataset1, self.dataset2, compat="minimal")
        self.assertIsInstance(result, xr.Dataset)

    def test_merge_handles_exception(self):
        dataset1_mock = MagicMock()
        dataset2_mock = MagicMock()
        dataset1_mock.__iter__ = MagicMock(return_value=iter([]))
        dataset2_mock.__iter__ = MagicMock(return_value=iter([]))

        with patch("xarray.merge") as mock_merge:
            mock_merge.side_effect = Exception("Merge failed")

            with self.assertRaises(DatacubeMergeError) as context:
                merge_datacubes(dataset1_mock, dataset2_mock)

            self.assertIn("Failed to merge datacubes", str(context.exception))
            self.assertIn("Merge failed", str(context.exception))

    def test_merge_with_overlapping_bands(self):
        dataset2_overlap = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(3, 10, 10)),
                "band3": (["time", "y", "x"], np.random.rand(3, 10, 10)),
            },
            coords={"time": self.dataset1.time, "y": self.dataset1.y, "x": self.dataset1.x},
        )

        result = merge_datacubes(self.dataset1, dataset2_overlap, compat="override")
        self.assertIn("band1", result.data_vars)
        self.assertIn("band3", result.data_vars)


class TestSelectTimeRange(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=10, freq="D")
        x = np.arange(10)
        y = np.arange(10)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(10, 10, 10)),
            },
            coords={"time": times, "y": y, "x": x},
        )

    def test_select_time_range_no_start_no_end(self):
        result = select_time_range(self.dataset)
        self.assertEqual(len(result.time), len(self.dataset.time))
        np.testing.assert_array_equal(result.time.values, self.dataset.time.values)

    def test_select_time_range_with_start_and_end(self):
        start = "2024-01-03"
        end = "2024-01-07"
        result = select_time_range(self.dataset, start=start, end=end)
        self.assertGreater(len(result.time), 0)
        self.assertLessEqual(result.time.min().values, pd.Timestamp(end))
        self.assertGreaterEqual(result.time.max().values, pd.Timestamp(start))

    def test_select_time_range_with_start_only(self):
        start = "2024-01-05"
        result = select_time_range(self.dataset, start=start)
        self.assertGreater(len(result.time), 0)
        self.assertGreaterEqual(result.time.min().values, pd.Timestamp(start))

    def test_select_time_range_with_end_only(self):
        end = "2024-01-05"
        result = select_time_range(self.dataset, end=end)
        self.assertGreater(len(result.time), 0)
        self.assertLessEqual(result.time.max().values, pd.Timestamp(end))

    def test_select_time_range_no_time_dimension_raises_error(self):
        dataset_no_time = xr.Dataset(
            {
                "band1": (["y", "x"], np.random.rand(10, 10)),
            },
            coords={"y": np.arange(10), "x": np.arange(10)},
        )

        with self.assertRaises(DatacubeOperationError) as context:
            select_time_range(dataset_no_time, start="2024-01-01")
        self.assertIn("Dataset does not have a time dimension", str(context.exception))

    def test_select_time_range_empty_result_raises_error(self):
        start = "2025-01-01"
        end = "2025-01-05"

        with self.assertRaises(DatacubeOperationError) as context:
            select_time_range(self.dataset, start=start, end=end)
        self.assertIn("No data found in time range", str(context.exception))
        self.assertIn("Available time range", str(context.exception))

    def test_select_time_range_invalid_time_raises_error(self):
        start = "invalid-date"

        with self.assertRaises(TypeError):
            select_time_range(self.dataset, start=start)

    def test_select_time_range_sorts_by_time(self):
        times_unsorted = (
            pd.date_range("2024-01-10", periods=5, freq="D").tolist()
            + pd.date_range("2024-01-01", periods=5, freq="D").tolist()
        )
        dataset_unsorted = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(10, 10, 10)),
            },
            coords={"time": times_unsorted, "y": np.arange(10), "x": np.arange(10)},
        )

        result = select_time_range(dataset_unsorted, start="2024-01-03", end="2024-01-07")
        time_index = result.time.to_index()
        self.assertTrue(time_index.is_monotonic_increasing)


class TestRechunkDatacube(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=3, freq="D")
        x = np.arange(100)
        y = np.arange(100)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(3, 100, 100)),
            },
            coords={"time": times, "y": y, "x": x},
        )

    def test_rechunk_with_dict(self):
        chunks = {"time": 1, "x": 50, "y": 50}
        result = rechunk_datacube(self.dataset, chunks)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_rechunk_with_auto_chunks(self):
        chunks = {"time": "auto", "x": "auto", "y": "auto"}
        result = rechunk_datacube(self.dataset, chunks)
        self.assertIsInstance(result, xr.Dataset)

    def test_rechunk_preserves_data(self):
        chunks = {"time": 1, "x": 25, "y": 25}
        result = rechunk_datacube(self.dataset, chunks)
        np.testing.assert_array_equal(result["band1"].values, self.dataset["band1"].values)

    def test_rechunk_preserves_coords(self):
        chunks = {"time": 1, "x": 50, "y": 50}
        result = rechunk_datacube(self.dataset, chunks)
        np.testing.assert_array_equal(result.time.values, self.dataset.time.values)
        np.testing.assert_array_equal(result.x.values, self.dataset.x.values)
        np.testing.assert_array_equal(result.y.values, self.dataset.y.values)

    def test_rechunk_with_partial_chunks(self):
        chunks = {"time": 1}
        result = rechunk_datacube(self.dataset, chunks)
        self.assertIsInstance(result, xr.Dataset)


if __name__ == "__main__":
    unittest.main()
