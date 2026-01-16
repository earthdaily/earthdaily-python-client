import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from earthdaily.datacube._temporal import temporal_aggregate, whittaker_smooth
from earthdaily.datacube.exceptions import DatacubeOperationError


class TestWhittakerSmooth(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=10, freq="D")
        x = np.arange(10)
        y = np.arange(10)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(10, 10, 10)),
                "band2": (["time", "y", "x"], np.random.rand(10, 10, 10)),
            },
            coords={"time": times, "y": y, "x": x},
        )

    @patch("earthdaily.datacube._temporal._whittaker_smooth_impl")
    def test_whittaker_smooth_with_defaults(self, mock_whittaker):
        mock_result = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(10, 10, 10)),
                "band2": (["time", "y", "x"], np.random.rand(10, 10, 10)),
            },
            coords=self.dataset.coords,
        )
        mock_whittaker.return_value = mock_result

        result = whittaker_smooth(self.dataset)

        self.assertIsInstance(result, xr.Dataset)
        self.assertEqual(mock_whittaker.call_count, 2)
        self.assertIn("band1", result.data_vars)
        self.assertIn("band2", result.data_vars)

    @patch("earthdaily.datacube._temporal._whittaker_smooth_impl")
    def test_whittaker_smooth_with_custom_beta(self, mock_whittaker):
        def mock_whittaker_side_effect(dataset, **kwargs):
            return dataset.copy()

        mock_whittaker.side_effect = mock_whittaker_side_effect

        custom_beta = 5000.0
        result = whittaker_smooth(self.dataset, beta=custom_beta)

        self.assertIsInstance(result, xr.Dataset)
        call_kwargs = mock_whittaker.call_args[1]
        self.assertEqual(call_kwargs["beta"], custom_beta)

    @patch("earthdaily.datacube._temporal._whittaker_smooth_impl")
    def test_whittaker_smooth_with_weights(self, mock_whittaker):
        def mock_whittaker_side_effect(dataset, **kwargs):
            return dataset.copy()

        mock_whittaker.side_effect = mock_whittaker_side_effect

        weights = np.ones(10)
        result = whittaker_smooth(self.dataset, weights=weights)

        self.assertIsInstance(result, xr.Dataset)
        call_kwargs = mock_whittaker.call_args[1]
        np.testing.assert_array_equal(call_kwargs["weights"], weights)

    @patch("earthdaily.datacube._temporal._whittaker_smooth_impl")
    def test_whittaker_smooth_with_custom_time_dim(self, mock_whittaker):
        def mock_whittaker_side_effect(dataset, **kwargs):
            return dataset.copy()

        mock_whittaker.side_effect = mock_whittaker_side_effect

        custom_time = "timestamp"
        result = whittaker_smooth(self.dataset, time=custom_time)

        self.assertIsInstance(result, xr.Dataset)
        call_kwargs = mock_whittaker.call_args[1]
        self.assertEqual(call_kwargs["time"], custom_time)

    @patch("earthdaily.datacube._temporal._whittaker_smooth_impl")
    def test_whittaker_smooth_applies_to_all_vars(self, mock_whittaker):
        mock_result_band1 = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(10, 10, 10)),
            },
            coords=self.dataset.coords,
        )
        mock_result_band2 = xr.Dataset(
            {
                "band2": (["time", "y", "x"], np.random.rand(10, 10, 10)),
            },
            coords=self.dataset.coords,
        )
        mock_whittaker.side_effect = [mock_result_band1, mock_result_band2]

        result = whittaker_smooth(self.dataset)

        self.assertEqual(mock_whittaker.call_count, 2)
        self.assertIn("band1", result.data_vars)
        self.assertIn("band2", result.data_vars)

    @patch("earthdaily.datacube._temporal._whittaker_smooth_impl")
    def test_whittaker_smooth_returns_copy(self, mock_whittaker):
        def mock_whittaker_side_effect(dataset, **kwargs):
            return dataset.copy()

        mock_whittaker.side_effect = mock_whittaker_side_effect

        original_values_band1 = self.dataset["band1"].values.copy()
        original_values_band2 = self.dataset["band2"].values.copy()
        result = whittaker_smooth(self.dataset)

        self.assertIsNot(result, self.dataset)
        np.testing.assert_array_equal(self.dataset["band1"].values, original_values_band1)
        np.testing.assert_array_equal(self.dataset["band2"].values, original_values_band2)


class TestTemporalAggregate(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=30, freq="D")
        x = np.arange(10)
        y = np.arange(10)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(30, 10, 10)),
                "band2": (["time", "y", "x"], np.random.rand(30, 10, 10)),
            },
            coords={"time": times, "y": y, "x": x},
        )

    def test_temporal_aggregate_mean(self):
        result = temporal_aggregate(self.dataset, method="mean", freq="1ME")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)
        self.assertIn("band2", result.data_vars)
        self.assertLess(len(result.time), len(self.dataset.time))

    def test_temporal_aggregate_median(self):
        result = temporal_aggregate(self.dataset, method="median", freq="1ME")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_temporal_aggregate_min(self):
        result = temporal_aggregate(self.dataset, method="min", freq="1ME")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_temporal_aggregate_max(self):
        result = temporal_aggregate(self.dataset, method="max", freq="1ME")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_temporal_aggregate_sum(self):
        result = temporal_aggregate(self.dataset, method="sum", freq="1ME")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_temporal_aggregate_std(self):
        result = temporal_aggregate(self.dataset, method="std", freq="1ME")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_temporal_aggregate_var(self):
        result = temporal_aggregate(self.dataset, method="var", freq="1ME")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_temporal_aggregate_with_default_freq(self):
        result = temporal_aggregate(self.dataset, method="mean")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_temporal_aggregate_no_time_dimension_raises_error(self):
        dataset_no_time = xr.Dataset(
            {
                "band1": (["y", "x"], np.random.rand(10, 10)),
            },
            coords={"y": np.arange(10), "x": np.arange(10)},
        )

        with self.assertRaises(DatacubeOperationError) as context:
            temporal_aggregate(dataset_no_time, method="mean")
        self.assertIn("Dataset does not have a time dimension", str(context.exception))

    def test_temporal_aggregate_unsupported_method_raises_error(self):
        with self.assertRaises(ValueError) as context:
            temporal_aggregate(self.dataset, method="invalid_method")
        self.assertIn("Unsupported aggregation method", str(context.exception))

    def test_temporal_aggregate_with_different_freq(self):
        result = temporal_aggregate(self.dataset, method="mean", freq="1W")
        self.assertIsInstance(result, xr.Dataset)
        self.assertLess(len(result.time), len(self.dataset.time))

    def test_temporal_aggregate_preserves_spatial_dims(self):
        result = temporal_aggregate(self.dataset, method="mean", freq="1ME")
        self.assertIn("y", result.dims)
        self.assertIn("x", result.dims)
        self.assertEqual(result.sizes["y"], self.dataset.sizes["y"])
        self.assertEqual(result.sizes["x"], self.dataset.sizes["x"])

    def test_temporal_aggregate_preserves_all_bands(self):
        result = temporal_aggregate(self.dataset, method="mean", freq="1ME")
        self.assertIn("band1", result.data_vars)
        self.assertIn("band2", result.data_vars)

    def test_temporal_aggregate_groupby_date(self):
        result = temporal_aggregate(self.dataset, method="mean", groupby="time.date")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)
        self.assertIn("time", result.dims)
        self.assertTrue(np.issubdtype(result.time.dtype, np.datetime64))

    def test_temporal_aggregate_groupby_unsupported(self):
        with self.assertRaises(ValueError):
            temporal_aggregate(self.dataset, method="mean", groupby="time.month")


if __name__ == "__main__":
    unittest.main()
