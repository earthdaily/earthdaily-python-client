import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from earthdaily.datacube._indices import add_indices


class TestAddIndices(unittest.TestCase):
    def setUp(self):
        self.dataset = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(3, 10, 10)),
                "nir": (["time", "y", "x"], np.random.rand(3, 10, 10)),
                "blue": (["time", "y", "x"], np.random.rand(3, 10, 10)),
            },
            coords={"time": np.arange(3), "y": np.arange(10), "x": np.arange(10)},
        )

    def test_no_kwargs_raises_error(self):
        with self.assertRaises(ValueError) as context:
            add_indices(self.dataset, ["NDVI"])
        self.assertIn("You must provide band parameters", str(context.exception))
        self.assertIn("spyndex documentation", str(context.exception))

    def test_empty_indices_list_raises_error(self):
        with self.assertRaises(ValueError) as context:
            add_indices(self.dataset, [], R=self.dataset["red"], N=self.dataset["nir"])
        self.assertIn("At least one index must be provided", str(context.exception))

    @patch("earthdaily.datacube._indices.spyndex.computeIndex")
    def test_single_index_success(self, mock_compute_index):
        mock_index_result = xr.DataArray(
            np.random.rand(3, 10, 10),
            dims=["time", "y", "x"],
            coords={"time": self.dataset.time, "y": self.dataset.y, "x": self.dataset.x},
        )
        mock_compute_index.return_value = mock_index_result

        result = add_indices(self.dataset, ["NDVI"], R=self.dataset["red"], N=self.dataset["nir"])

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("NDVI", result.data_vars)
        self.assertEqual(mock_compute_index.call_count, 1)
        call_kwargs = mock_compute_index.call_args[1]
        self.assertEqual(call_kwargs["index"], ["NDVI"])
        self.assertIn("R", call_kwargs["params"])
        self.assertIn("N", call_kwargs["params"])

    @patch("earthdaily.datacube._indices.spyndex.computeIndex")
    def test_multiple_indices_success(self, mock_compute_index):
        mock_index_result = xr.DataArray(
            np.random.rand(2, 3, 10, 10),
            dims=["index", "time", "y", "x"],
            coords={"index": ["NDVI", "EVI"], "time": self.dataset.time, "y": self.dataset.y, "x": self.dataset.x},
        )
        mock_compute_index.return_value = mock_index_result

        result = add_indices(self.dataset, ["NDVI", "EVI"], R=self.dataset["red"], N=self.dataset["nir"])

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("NDVI", result.data_vars)
        self.assertIn("EVI", result.data_vars)
        self.assertEqual(mock_compute_index.call_count, 1)
        call_kwargs = mock_compute_index.call_args[1]
        self.assertEqual(call_kwargs["index"], ["NDVI", "EVI"])
        self.assertIn("R", call_kwargs["params"])
        self.assertIn("N", call_kwargs["params"])

    @patch("earthdaily.datacube._indices.spyndex.computeIndex")
    def test_single_index_expands_dims(self, mock_compute_index):
        mock_index_result = xr.DataArray(
            np.random.rand(3, 10, 10),
            dims=["time", "y", "x"],
            coords={"time": self.dataset.time, "y": self.dataset.y, "x": self.dataset.x},
        )
        mock_compute_index.return_value = mock_index_result

        result = add_indices(self.dataset, ["NDVI"], R=self.dataset["red"], N=self.dataset["nir"])

        self.assertIn("NDVI", result.data_vars)
        ndvi_var = result["NDVI"]
        self.assertIsInstance(ndvi_var, xr.DataArray)

    @patch("earthdaily.datacube._indices.spyndex.computeIndex")
    def test_original_bands_preserved(self, mock_compute_index):
        mock_index_result = xr.DataArray(
            np.random.rand(3, 10, 10),
            dims=["time", "y", "x"],
            coords={"time": self.dataset.time, "y": self.dataset.y, "x": self.dataset.x},
        )
        mock_compute_index.return_value = mock_index_result

        result = add_indices(self.dataset, ["NDVI"], R=self.dataset["red"], N=self.dataset["nir"])

        self.assertIn("red", result.data_vars)
        self.assertIn("nir", result.data_vars)
        self.assertIn("blue", result.data_vars)

    @patch("earthdaily.datacube._indices.spyndex.computeIndex")
    def test_spyndex_exception_raises_value_error(self, mock_compute_index):
        mock_compute_index.side_effect = Exception("spyndex error message")

        with self.assertRaises(ValueError) as context:
            add_indices(self.dataset, ["NDVI"], R=self.dataset["red"], N=self.dataset["nir"])

        error_msg = str(context.exception)
        self.assertIn("Failed to compute indices", error_msg)
        self.assertIn("spyndex error message", error_msg)
        self.assertIn("Available bands in datacube", error_msg)
        self.assertIn("Provided parameters", error_msg)
        self.assertIn("red", error_msg)
        self.assertIn("nir", error_msg)
        self.assertIn("blue", error_msg)
        self.assertIn("R", error_msg)
        self.assertIn("N", error_msg)

    @patch("earthdaily.datacube._indices.spyndex.computeIndex")
    def test_custom_parameters_passed(self, mock_compute_index):
        mock_index_result = xr.DataArray(
            np.random.rand(3, 10, 10),
            dims=["time", "y", "x"],
            coords={"time": self.dataset.time, "y": self.dataset.y, "x": self.dataset.x},
        )
        mock_compute_index.return_value = mock_index_result

        custom_params = {
            "R": self.dataset["red"],
            "N": self.dataset["nir"],
            "B": self.dataset["blue"],
            "alpha": 2.0,
        }

        result = add_indices(self.dataset, ["NDVI"], **custom_params)

        self.assertIsInstance(result, xr.Dataset)
        call_kwargs = mock_compute_index.call_args[1]
        self.assertIn("R", call_kwargs["params"])
        self.assertIn("N", call_kwargs["params"])
        self.assertIn("B", call_kwargs["params"])
        self.assertEqual(call_kwargs["params"]["alpha"], 2.0)

    @patch("earthdaily.datacube._indices.spyndex.computeIndex")
    def test_dataset_copy_not_mutated(self, mock_compute_index):
        mock_index_result = xr.DataArray(
            np.random.rand(3, 10, 10),
            dims=["time", "y", "x"],
            coords={"time": self.dataset.time, "y": self.dataset.y, "x": self.dataset.x},
        )
        mock_compute_index.return_value = mock_index_result

        original_bands = set(self.dataset.data_vars.keys())
        result = add_indices(self.dataset, ["NDVI"], R=self.dataset["red"], N=self.dataset["nir"])

        self.assertEqual(set(self.dataset.data_vars.keys()), original_bands)
        self.assertGreater(len(result.data_vars), len(self.dataset.data_vars))

    @patch("earthdaily.datacube._indices.spyndex.computeIndex")
    def test_empty_dataset_with_indices(self, mock_compute_index):
        empty_dataset = xr.Dataset()
        mock_index_result = xr.DataArray(np.random.rand(3, 10, 10), dims=["time", "y", "x"])
        mock_compute_index.return_value = mock_index_result

        result = add_indices(
            empty_dataset,
            ["NDVI"],
            R=xr.DataArray(np.random.rand(3, 10, 10)),
            N=xr.DataArray(np.random.rand(3, 10, 10)),
        )

        self.assertIn("NDVI", result.data_vars)


if __name__ == "__main__":
    unittest.main()
