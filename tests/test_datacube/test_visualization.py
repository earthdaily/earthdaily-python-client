import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from earthdaily.datacube._visualization import _max_time_wrap, plot_band, plot_mask, plot_rgb, thumbnail
from earthdaily.datacube.exceptions import DatacubeVisualizationError


class TestMaxTimeWrap(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=10, freq="D")
        self.dataset_with_time = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(10, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )

        self.dataset_without_time = xr.Dataset(
            {
                "band1": (["y", "x"], np.random.rand(10, 10)),
            },
            coords={"y": np.arange(10), "x": np.arange(10)},
        )

    def test_max_time_wrap_with_time_dimension_less_than_wish(self):
        wish = 15
        result = _max_time_wrap(self.dataset_with_time, wish=wish)
        self.assertEqual(result, 10)

    def test_max_time_wrap_with_time_dimension_greater_than_wish(self):
        wish = 5
        result = _max_time_wrap(self.dataset_with_time, wish=wish)
        self.assertEqual(result, 5)

    def test_max_time_wrap_with_time_dimension_equal_to_wish(self):
        wish = 10
        result = _max_time_wrap(self.dataset_with_time, wish=wish)
        self.assertEqual(result, 10)

    def test_max_time_wrap_without_time_dimension(self):
        wish = 5
        result = _max_time_wrap(self.dataset_without_time, wish=wish)
        self.assertEqual(result, 5)

    def test_max_time_wrap_with_custom_col(self):
        dataset_with_custom_dim = xr.Dataset(
            {
                "band1": (["custom_dim", "y", "x"], np.random.rand(7, 10, 10)),
            },
            coords={"custom_dim": np.arange(7), "y": np.arange(10), "x": np.arange(10)},
        )
        wish = 10
        result = _max_time_wrap(dataset_with_custom_dim, wish=wish, col="custom_dim")
        self.assertEqual(result, 7)


class TestPlotRgb(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        self.dataset = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "green": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "blue": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_rgb_with_defaults(self, mock_plot):
        mock_plot.return_value = MagicMock()
        result = plot_rgb(self.dataset)
        self.assertIsNotNone(result)

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_rgb_with_custom_bands(self, mock_plot):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        dataset_custom = xr.Dataset(
            {
                "R": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "G": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "B": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )
        mock_plot.return_value = MagicMock()
        result = plot_rgb(dataset_custom, red="R", green="G", blue="B")
        self.assertIsNotNone(result)

    def test_plot_rgb_red_band_not_found(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        dataset_missing = xr.Dataset(
            {
                "green": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "blue": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )
        with self.assertRaises(DatacubeVisualizationError) as context:
            plot_rgb(dataset_missing)
        self.assertIn("Band 'red' not found", str(context.exception))

    def test_plot_rgb_green_band_not_found(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        dataset_missing = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "blue": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )
        with self.assertRaises(DatacubeVisualizationError) as context:
            plot_rgb(dataset_missing)
        self.assertIn("Band 'green' not found", str(context.exception))

    def test_plot_rgb_blue_band_not_found(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        dataset_missing = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "green": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )
        with self.assertRaises(DatacubeVisualizationError) as context:
            plot_rgb(dataset_missing)
        self.assertIn("Band 'blue' not found", str(context.exception))

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_rgb_with_background(self, mock_plot):
        dataset_with_nan = self.dataset.copy()
        dataset_with_nan["red"].values[0, 0, 0] = np.nan
        mock_plot.return_value = MagicMock()
        result = plot_rgb(dataset_with_nan, background=0)
        self.assertIsNotNone(result)

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_rgb_with_custom_col_wrap(self, mock_plot):
        mock_plot.return_value = MagicMock()
        result = plot_rgb(self.dataset, col_wrap=3)
        self.assertIsNotNone(result)

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_rgb_with_kwargs(self, mock_plot):
        mock_plot.return_value = MagicMock()
        result = plot_rgb(self.dataset, figsize=(10, 10))
        self.assertIsNotNone(result)


class TestPlotBand(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "band2": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_band_with_defaults(self, mock_plot):
        mock_plot.return_value = MagicMock()
        result = plot_band(self.dataset, "band1")
        self.assertIsNotNone(result)

    def test_plot_band_not_found(self):
        with self.assertRaises(DatacubeVisualizationError) as context:
            plot_band(self.dataset, "nonexistent")
        self.assertIn("Band 'nonexistent' not found", str(context.exception))

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_band_with_custom_cmap(self, mock_plot):
        mock_plot.return_value = MagicMock()
        result = plot_band(self.dataset, "band1", cmap="viridis")
        self.assertIsNotNone(result)

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_band_with_custom_col_wrap(self, mock_plot):
        mock_plot.return_value = MagicMock()
        result = plot_band(self.dataset, "band1", col_wrap=3)
        self.assertIsNotNone(result)

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_band_with_kwargs(self, mock_plot):
        mock_plot.return_value = MagicMock()
        result = plot_band(self.dataset, "band1", figsize=(10, 10))
        self.assertIsNotNone(result)


class TestPlotMask(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=3, freq="D")
        self.dataset = xr.Dataset(
            {
                "cloud-mask": (["time", "y", "x"], np.random.randint(0, 4, size=(3, 5, 5))),
                "other": (["time", "y", "x"], np.random.rand(3, 5, 5)),
            },
            coords={"time": times, "y": np.arange(5), "x": np.arange(5)},
        )

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_mask_from_dataset(self, mock_plot):
        mock_plot.return_value = MagicMock()
        result = plot_mask(self.dataset)
        self.assertIsNotNone(result)

    def test_plot_mask_band_not_found(self):
        with self.assertRaises(DatacubeVisualizationError):
            plot_mask(self.dataset.drop_vars("cloud-mask"))

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_plot_mask_from_data_array(self, mock_plot):
        mock_plot.return_value = MagicMock()
        mask_array = self.dataset["cloud-mask"]
        result = plot_mask(mask_array, cmap="viridis", col_wrap=2)
        self.assertIsNotNone(result)


class TestThumbnail(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        self.dataset = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "green": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "blue": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_thumbnail_with_defaults(self, mock_plot):
        mock_imshow = MagicMock()
        mock_axes = MagicMock()
        mock_figure = MagicMock(spec=Figure)
        mock_axes.figure = mock_figure
        mock_imshow.return_value = mock_axes
        mock_plot.imshow = mock_imshow

        result = thumbnail(self.dataset)
        self.assertIsNotNone(result)

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_thumbnail_with_custom_time_index(self, mock_plot):
        mock_imshow = MagicMock()
        mock_axes = MagicMock()
        mock_figure = MagicMock(spec=Figure)
        mock_axes.figure = mock_figure
        mock_imshow.return_value = mock_axes
        mock_plot.imshow = mock_imshow

        result = thumbnail(self.dataset, time_index=2)
        self.assertIsNotNone(result)

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_thumbnail_without_time_dimension(self, mock_plot):
        dataset_no_time = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(10, 10)),
                "green": (["y", "x"], np.random.rand(10, 10)),
                "blue": (["y", "x"], np.random.rand(10, 10)),
            },
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        mock_imshow = MagicMock()
        mock_axes = MagicMock()
        mock_figure = MagicMock(spec=Figure)
        mock_axes.figure = mock_figure
        mock_imshow.return_value = mock_axes
        mock_plot.imshow = mock_imshow

        result = thumbnail(dataset_no_time)
        self.assertIsNotNone(result)

    def test_thumbnail_red_band_not_found(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        dataset_missing = xr.Dataset(
            {
                "green": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "blue": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )
        with self.assertRaises(DatacubeVisualizationError) as context:
            thumbnail(dataset_missing)
        self.assertIn("Band 'red' not found", str(context.exception))

    def test_thumbnail_green_band_not_found(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        dataset_missing = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "blue": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )
        with self.assertRaises(DatacubeVisualizationError) as context:
            thumbnail(dataset_missing)
        self.assertIn("Band 'green' not found", str(context.exception))

    def test_thumbnail_blue_band_not_found(self):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        dataset_missing = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "green": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )
        with self.assertRaises(DatacubeVisualizationError) as context:
            thumbnail(dataset_missing)
        self.assertIn("Band 'blue' not found", str(context.exception))

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_thumbnail_time_index_out_of_bounds(self, mock_plot):
        mock_imshow = MagicMock()
        mock_axes = MagicMock()
        mock_figure = MagicMock(spec=Figure)
        mock_axes.figure = mock_figure
        mock_imshow.return_value = mock_axes
        mock_plot.imshow = mock_imshow

        result = thumbnail(self.dataset, time_index=10)
        self.assertIsNotNone(result)

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_thumbnail_with_custom_bands(self, mock_plot):
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        dataset_custom = xr.Dataset(
            {
                "R": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "G": (["time", "y", "x"], np.random.rand(5, 10, 10)),
                "B": (["time", "y", "x"], np.random.rand(5, 10, 10)),
            },
            coords={"time": times, "y": np.arange(10), "x": np.arange(10)},
        )
        mock_imshow = MagicMock()
        mock_axes = MagicMock()
        mock_figure = MagicMock(spec=Figure)
        mock_axes.figure = mock_figure
        mock_imshow.return_value = mock_axes
        mock_plot.imshow = mock_imshow

        result = thumbnail(dataset_custom, red="R", green="G", blue="B")
        self.assertIsNotNone(result)

    @patch("earthdaily.datacube._visualization.xr.DataArray.plot")
    def test_thumbnail_with_kwargs(self, mock_plot):
        mock_imshow = MagicMock()
        mock_axes = MagicMock()
        mock_figure = MagicMock(spec=Figure)
        mock_axes.figure = mock_figure
        mock_imshow.return_value = mock_axes
        mock_plot.imshow = mock_imshow

        result = thumbnail(self.dataset, figsize=(10, 10))
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
