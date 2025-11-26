import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from earthdaily.datacube._datacube import Datacube


class TestDatacubeInit(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        x = np.arange(10)
        y = np.arange(10)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "band2": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": y, "x": x},
        )

    def test_init_with_dataset(self):
        dc = Datacube(self.dataset)
        self.assertIsInstance(dc, Datacube)
        self.assertEqual(dc._dataset, self.dataset)

    def test_init_with_metadata(self):
        metadata = {"test": "value", "count": 42}
        dc = Datacube(self.dataset, metadata)
        self.assertEqual(dc._metadata, metadata)

    def test_init_without_metadata(self):
        dc = Datacube(self.dataset)
        self.assertEqual(dc._metadata, {})

    def test_init_with_none_metadata(self):
        dc = Datacube(self.dataset, None)
        self.assertEqual(dc._metadata, {})


class TestDatacubeProperties(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(5, 20, 20)),
                "band2": (["time", "y", "x"], np.random.rand(5, 20, 20)),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.dataset.rio.write_crs("EPSG:4326", inplace=True)
        self.dataset.rio.write_transform(inplace=True)

        self.dc = Datacube(self.dataset)

    def test_data_property(self):
        self.assertIsInstance(self.dc.data, xr.Dataset)
        self.assertEqual(self.dc.data, self.dataset)

    def test_bands_property(self):
        bands = self.dc.bands
        self.assertIsInstance(bands, list)
        self.assertIn("band1", bands)
        self.assertIn("band2", bands)

    def test_timestamps_property(self):
        timestamps = self.dc.timestamps
        self.assertIsInstance(timestamps, list)
        self.assertEqual(len(timestamps), 5)

    def test_timestamps_property_no_time_dim(self):
        dataset_no_time = xr.Dataset(
            {
                "band1": (["y", "x"], np.random.rand(10, 10)),
            },
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        dc = Datacube(dataset_no_time)
        self.assertEqual(dc.timestamps, [])

    def test_timestamps_property_empty_time(self):
        dataset_empty_time = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(0, 10, 10)),
            },
            coords={"time": pd.DatetimeIndex([]), "y": np.arange(10), "x": np.arange(10)},
        )
        dc = Datacube(dataset_empty_time)
        self.assertEqual(dc.timestamps, [])

    def test_crs_property(self):
        crs = self.dc.crs
        self.assertIsNotNone(crs)
        self.assertIn("4326", crs)

    def test_crs_property_no_crs(self):
        dataset_no_crs = xr.Dataset(
            {
                "band1": (["y", "x"], np.random.rand(10, 10)),
            },
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        dc = Datacube(dataset_no_crs)
        self.assertIn(dc.crs, [None, "None"])

    def test_resolution_property(self):
        resolution = self.dc.resolution
        self.assertIsNotNone(resolution)
        self.assertIsInstance(resolution, tuple)
        self.assertEqual(len(resolution), 2)

    def test_resolution_property_no_crs(self):
        dataset_no_crs = xr.Dataset(
            {
                "band1": (["y", "x"], np.random.rand(10, 10)),
            },
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        dc = Datacube(dataset_no_crs)
        resolution = dc.resolution
        self.assertIsNotNone(resolution)
        self.assertIsInstance(resolution, tuple)

    def test_shape_property(self):
        shape = self.dc.shape
        self.assertIsInstance(shape, dict)
        self.assertIn("time", shape)
        self.assertIn("y", shape)
        self.assertIn("x", shape)

    def test_extent_property(self):
        extent = self.dc.extent
        self.assertIsNotNone(extent)
        self.assertIsInstance(extent, tuple)
        self.assertEqual(len(extent), 4)

    def test_extent_property_no_crs(self):
        dataset_no_crs = xr.Dataset(
            {
                "band1": (["y", "x"], np.random.rand(10, 10)),
            },
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        dc = Datacube(dataset_no_crs)
        extent = dc.extent
        self.assertIsNotNone(extent)
        self.assertIsInstance(extent, tuple)
        self.assertEqual(len(extent), 4)


class TestDatacubeMethods(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(5, 20, 20)),
                "band2": (["time", "y", "x"], np.random.rand(5, 20, 20)),
                "red": (["time", "y", "x"], np.random.rand(5, 20, 20)),
                "green": (["time", "y", "x"], np.random.rand(5, 20, 20)),
                "blue": (["time", "y", "x"], np.random.rand(5, 20, 20)),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.dataset.rio.write_crs("EPSG:4326", inplace=True)
        self.dataset.rio.write_transform(inplace=True)

        self.dc = Datacube(self.dataset, {"test": "metadata"})

    @patch("earthdaily.datacube._datacube.apply_cloud_mask")
    def test_apply_mask(self, mock_apply_mask):
        mock_result = self.dataset.copy()
        mock_apply_mask.return_value = mock_result

        result = self.dc.apply_mask("band1", clear_values=[1])

        self.assertIsInstance(result, Datacube)
        self.assertNotEqual(result, self.dc)
        mock_apply_mask.assert_called_once()

    @patch("earthdaily.datacube._datacube.clip_datacube")
    def test_clip(self, mock_clip):
        geometry = gpd.GeoDataFrame(geometry=[box(2, 2, 8, 8)], crs="EPSG:4326")
        mock_result = self.dataset.copy()
        mock_clip.return_value = mock_result

        result = self.dc.clip(geometry)

        self.assertIsInstance(result, Datacube)
        mock_clip.assert_called_once_with(self.dataset, geometry)

    @patch("earthdaily.datacube._datacube.merge_datacubes")
    def test_merge(self, mock_merge):
        other_dataset = xr.Dataset(
            {
                "band3": (["time", "y", "x"], np.random.rand(5, 20, 20)),
            },
            coords={"time": self.dataset.time, "y": self.dataset.y, "x": self.dataset.x},
        )
        other_dc = Datacube(other_dataset)
        mock_result = xr.merge([self.dataset, other_dataset], compat="override")
        mock_merge.return_value = mock_result

        result = self.dc.merge(other_dc)

        self.assertIsInstance(result, Datacube)
        mock_merge.assert_called_once_with(self.dataset, other_dataset, compat="override")

    def test_merge_with_non_datacube_raises_error(self):
        with self.assertRaises(TypeError) as context:
            self.dc.merge("not a datacube")
        self.assertIn("Can only merge with another Datacube instance", str(context.exception))

    @patch("earthdaily.datacube._datacube.select_time_range")
    def test_select_time(self, mock_select_time):
        mock_result = self.dataset.copy()
        mock_select_time.return_value = mock_result

        result = self.dc.select_time(start="2024-01-02", end="2024-01-04")

        self.assertIsInstance(result, Datacube)
        mock_select_time.assert_called_once_with(self.dataset, "2024-01-02", "2024-01-04")

    @patch("earthdaily.datacube._datacube.rechunk_datacube")
    def test_rechunk(self, mock_rechunk):
        chunks = {"time": 1, "x": 10, "y": 10}
        mock_result = self.dataset.copy()
        mock_rechunk.return_value = mock_result

        result = self.dc.rechunk(chunks)

        self.assertIsInstance(result, Datacube)
        mock_rechunk.assert_called_once_with(self.dataset, chunks)

    @patch("earthdaily.datacube._datacube.plot_rgb")
    def test_plot_rgb(self, mock_plot_rgb):
        mock_plot_rgb.return_value = MagicMock()
        result = self.dc.plot_rgb()
        self.assertIsNotNone(result)
        mock_plot_rgb.assert_called_once()

    @patch("earthdaily.datacube._datacube.plot_band")
    def test_plot_band(self, mock_plot_band):
        mock_plot_band.return_value = MagicMock()
        result = self.dc.plot_band("band1")
        self.assertIsNotNone(result)
        mock_plot_band.assert_called_once()

    @patch("earthdaily.datacube._datacube.thumbnail")
    def test_thumbnail(self, mock_thumbnail):
        mock_figure = MagicMock()
        mock_thumbnail.return_value = mock_figure
        result = self.dc.thumbnail()
        self.assertEqual(result, mock_figure)
        mock_thumbnail.assert_called_once()

    @patch("earthdaily.datacube._datacube.add_indices")
    def test_add_indices(self, mock_add_indices):
        mock_result = self.dataset.copy()
        mock_add_indices.return_value = mock_result

        result = self.dc.add_indices(["NDVI"], R=self.dataset["red"], N=self.dataset["band1"])

        self.assertIsInstance(result, Datacube)
        mock_add_indices.assert_called_once()

    @patch("earthdaily.datacube._datacube.compute_zonal_stats")
    def test_zonal_stats(self, mock_zonal_stats):
        geometry = gpd.GeoDataFrame(geometry=[box(2, 2, 8, 8)], crs="EPSG:4326")
        mock_result = xr.Dataset()
        mock_zonal_stats.return_value = mock_result

        result = self.dc.zonal_stats(geometry)

        self.assertIsNotNone(result)
        mock_zonal_stats.assert_called_once()

    @patch("earthdaily.datacube._datacube.whittaker_smooth")
    def test_whittaker(self, mock_whittaker):
        mock_result = self.dataset.copy()
        mock_whittaker.return_value = mock_result

        result = self.dc.whittaker()

        self.assertIsInstance(result, Datacube)
        mock_whittaker.assert_called_once()

    @patch("earthdaily.datacube._datacube.temporal_aggregate")
    def test_temporal_aggregate(self, mock_temporal_aggregate):
        mock_result = self.dataset.copy()
        mock_temporal_aggregate.return_value = mock_result

        result = self.dc.temporal_aggregate()

        self.assertIsInstance(result, Datacube)
        mock_temporal_aggregate.assert_called_once()

    @patch("earthdaily.datacube._datacube.temporal_aggregate")
    def test_resample(self, mock_temporal_aggregate):
        mock_result = self.dataset.copy()
        mock_temporal_aggregate.return_value = mock_result

        result = self.dc.resample("1W")

        self.assertIsInstance(result, Datacube)
        mock_temporal_aggregate.assert_called_once()


class TestDatacubeCreateNew(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        x = np.arange(10)
        y = np.arange(10)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.metadata = {"original": "test", "count": 5}
        self.dc = Datacube(self.dataset, self.metadata)

    def test_create_new_preserves_metadata(self):
        new_dataset = self.dataset.copy()
        new_dataset["band2"] = (["time", "y", "x"], np.random.rand(5, 10, 10))
        result = self.dc._create_new(new_dataset)

        self.assertIsInstance(result, Datacube)
        self.assertEqual(result._metadata["original"], self.metadata["original"])
        self.assertEqual(result._metadata["count"], self.metadata["count"])

    def test_create_new_updates_metadata(self):
        new_dataset = self.dataset.copy()
        result = self.dc._create_new(new_dataset, count=10, new_key="new_value")

        self.assertEqual(result._metadata["count"], 10)
        self.assertEqual(result._metadata["new_key"], "new_value")
        self.assertEqual(result._metadata["original"], self.metadata["original"])

    def test_create_new_does_not_modify_original(self):
        new_dataset = self.dataset.copy()
        original_metadata = self.dc._metadata.copy()
        self.dc._create_new(new_dataset, test="value")

        self.assertEqual(self.dc._metadata, original_metadata)


class TestDatacubeInfo(unittest.TestCase):
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

        self.dc = Datacube(self.dataset)

    def test_info(self):
        info_str = self.dc.info()
        self.assertIsInstance(info_str, str)
        self.assertIn("Datacube Information", info_str)
        self.assertIn("Dimensions", info_str)
        self.assertIn("Bands", info_str)
        self.assertIn("band1", info_str)
        self.assertIn("band2", info_str)

    def test_info_with_timestamps(self):
        info_str = self.dc.info()
        self.assertIn("Timestamps", info_str)
        self.assertIn("time steps", info_str)
        self.assertIn("Time range", info_str)

    def test_info_without_timestamps(self):
        dataset_no_time = xr.Dataset(
            {
                "band1": (["y", "x"], np.random.rand(10, 10)),
            },
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        dataset_no_time.rio.write_crs("EPSG:4326", inplace=True)
        dataset_no_time.rio.write_transform(inplace=True)
        dc = Datacube(dataset_no_time)
        info_str = dc.info()
        self.assertIn("Datacube Information", info_str)
        self.assertNotIn("Time range", info_str)


class TestDatacubeRepr(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        x = np.arange(10)
        y = np.arange(10)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "band2": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.dataset.rio.write_crs("EPSG:4326", inplace=True)
        self.dataset.rio.write_transform(inplace=True)

        self.dc = Datacube(self.dataset)

    def test_repr(self):
        repr_str = repr(self.dc)
        self.assertIsInstance(repr_str, str)
        self.assertIn("Datacube", repr_str)
        self.assertIn("bands=", repr_str)
        self.assertIn("shape=", repr_str)
        self.assertIn("crs=", repr_str)

    def test_str(self):
        str_str = str(self.dc)
        self.assertIsInstance(str_str, str)
        self.assertIn("Datacube Information", str_str)


if __name__ == "__main__":
    unittest.main()
