import unittest

import numpy as np
import pandas as pd
import xarray as xr

from earthdaily.datacube._whittaker import _create_banded_matrix, _whittaker_pixel, whittaker_smooth
from earthdaily.datacube.constants import DEFAULT_WHITTAKER_BETA, WHITTAKER_ALPHA


class TestWhittakerSmooth(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01", periods=10, freq="2D")
        x = np.arange(5)
        y = np.arange(5)

        self.dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(10, 5, 5)),
                "band2": (["time", "y", "x"], np.random.rand(10, 5, 5)),
            },
            coords={"time": times, "y": y, "x": x},
        )

    def test_whittaker_smooth_with_defaults(self):
        result = whittaker_smooth(self.dataset)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)
        self.assertIn("band2", result.data_vars)
        self.assertEqual(len(result.time), len(self.dataset.time))

    def test_whittaker_smooth_with_custom_beta(self):
        custom_beta = 5000.0
        result = whittaker_smooth(self.dataset, beta=custom_beta)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_whittaker_smooth_with_weights(self):
        weights = np.ones(10)
        result = whittaker_smooth(self.dataset, weights=weights)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_whittaker_smooth_with_scalar_weight(self):
        result = whittaker_smooth(self.dataset, weights=0.5)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_whittaker_smooth_with_invalid_weight_length_raises(self):
        with self.assertRaises(ValueError):
            whittaker_smooth(self.dataset, weights=np.ones(5))

    def test_whittaker_smooth_with_weights_list(self):
        weights = [1.0] * 10
        result = whittaker_smooth(self.dataset, weights=weights)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_whittaker_smooth_with_custom_time_dim(self):
        dataset_custom = xr.Dataset(
            {
                "band1": (["timestamp", "y", "x"], np.random.rand(10, 5, 5)),
            },
            coords={
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="2D"),
                "y": np.arange(5),
                "x": np.arange(5),
            },
        )
        result = whittaker_smooth(dataset_custom, time="timestamp")
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_whittaker_smooth_preserves_shape(self):
        result = whittaker_smooth(self.dataset)
        self.assertEqual(result.sizes["time"], self.dataset.sizes["time"])
        self.assertEqual(result.sizes["y"], self.dataset.sizes["y"])
        self.assertEqual(result.sizes["x"], self.dataset.sizes["x"])

    def test_whittaker_smooth_with_nan_values(self):
        dataset_with_nan = self.dataset.copy()
        dataset_with_nan["band1"].values[0, 0, 0] = np.nan
        result = whittaker_smooth(dataset_with_nan)
        self.assertIsInstance(result, xr.Dataset)

    def test_whittaker_smooth_single_band(self):
        dataset_single = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(10, 5, 5)),
            },
            coords={"time": self.dataset.time, "y": np.arange(5), "x": np.arange(5)},
        )
        result = whittaker_smooth(dataset_single)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("band1", result.data_vars)

    def test_whittaker_smooth_different_frequencies(self):
        times_weekly = pd.date_range("2024-01-01", periods=5, freq="7D")
        dataset_weekly = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(5, 5, 5)),
            },
            coords={"time": times_weekly, "y": np.arange(5), "x": np.arange(5)},
        )
        result = whittaker_smooth(dataset_weekly)
        self.assertIsInstance(result, xr.Dataset)


class TestCreateBandedMatrix(unittest.TestCase):
    def test_create_banded_matrix_shape(self):
        m = 10
        beta = 1000.0
        result = _create_banded_matrix(m, beta)
        self.assertEqual(result.shape, (2 * WHITTAKER_ALPHA + 1, m))

    def test_create_banded_matrix_with_default_beta(self):
        m = 10
        result = _create_banded_matrix(m, DEFAULT_WHITTAKER_BETA)
        self.assertEqual(result.shape, (2 * WHITTAKER_ALPHA + 1, m))

    def test_create_banded_matrix_scales_with_beta(self):
        m = 10
        beta1 = 1000.0
        beta2 = 2000.0
        result1 = _create_banded_matrix(m, beta1)
        result2 = _create_banded_matrix(m, beta2)
        np.testing.assert_array_almost_equal(result2, result1 * 2)

    def test_create_banded_matrix_small_m(self):
        m = 5
        beta = 1000.0
        result = _create_banded_matrix(m, beta)
        self.assertEqual(result.shape, (2 * WHITTAKER_ALPHA + 1, m))

    def test_create_banded_matrix_large_m(self):
        m = 100
        beta = 1000.0
        result = _create_banded_matrix(m, beta)
        self.assertEqual(result.shape, (2 * WHITTAKER_ALPHA + 1, m))

    def test_create_banded_matrix_zero_beta(self):
        m = 10
        beta = 0.0
        result = _create_banded_matrix(m, beta)
        self.assertEqual(result.shape, (2 * WHITTAKER_ALPHA + 1, m))
        np.testing.assert_array_equal(result, np.zeros((2 * WHITTAKER_ALPHA + 1, m)))

    def test_create_banded_matrix_structure(self):
        m = 10
        beta = 1.0
        result = _create_banded_matrix(m, beta)
        self.assertEqual(result.shape[0], 2 * WHITTAKER_ALPHA + 1)
        self.assertEqual(result.shape[1], m)


class TestWhittakerPixel(unittest.TestCase):
    def setUp(self):
        self.m = 10
        self.ab_mat = _create_banded_matrix(self.m, DEFAULT_WHITTAKER_BETA)

    def test_whittaker_pixel_basic(self):
        signal = np.random.rand(self.m)
        weights = np.ones(self.m)
        result = _whittaker_pixel(signal, weights, self.ab_mat)
        self.assertEqual(len(result), self.m)
        self.assertIsInstance(result, np.ndarray)

    def test_whittaker_pixel_with_nan(self):
        signal = np.random.rand(self.m)
        signal[0] = np.nan
        weights = np.ones(self.m)
        result = _whittaker_pixel(signal, weights, self.ab_mat)
        self.assertEqual(len(result), self.m)

    def test_whittaker_pixel_all_nan(self):
        signal = np.full(self.m, np.nan)
        weights = np.ones(self.m)
        result = _whittaker_pixel(signal, weights, self.ab_mat)
        np.testing.assert_array_equal(result, signal)

    def test_whittaker_pixel_with_custom_weights(self):
        signal = np.random.rand(self.m)
        weights = np.random.rand(self.m)
        result = _whittaker_pixel(signal, weights, self.ab_mat)
        self.assertEqual(len(result), self.m)

    def test_whittaker_pixel_with_zero_weights(self):
        signal = np.random.rand(self.m)
        weights = np.zeros(self.m)
        result = _whittaker_pixel(signal, weights, self.ab_mat)
        self.assertEqual(len(result), self.m)

    def test_whittaker_pixel_preserves_shape(self):
        signal = np.random.rand(self.m)
        weights = np.ones(self.m)
        result = _whittaker_pixel(signal, weights, self.ab_mat)
        self.assertEqual(result.shape, signal.shape)

    def test_whittaker_pixel_small_signal(self):
        m_small = 5
        ab_mat_small = _create_banded_matrix(m_small, DEFAULT_WHITTAKER_BETA)
        signal = np.random.rand(m_small)
        weights = np.ones(m_small)
        result = _whittaker_pixel(signal, weights, ab_mat_small)
        self.assertEqual(len(result), m_small)

    def test_whittaker_pixel_does_not_modify_input(self):
        signal = np.random.rand(self.m)
        signal_original = signal.copy()
        weights = np.ones(self.m)
        _whittaker_pixel(signal, weights, self.ab_mat)
        np.testing.assert_array_equal(signal, signal_original)


if __name__ == "__main__":
    unittest.main()
