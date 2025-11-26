import unittest

import dask.array as da
import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from earthdaily.datacube import apply_cloud_mask
from earthdaily.datacube._masking import _resolve_geometry_selection
from earthdaily.datacube.exceptions import DatacubeMaskingError


class TestResolveGeometrySelection(unittest.TestCase):
    def test_bbox_precedence_over_intersects(self):
        bbox = [0.0, 0.0, 1.0, 1.0]
        intersects = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs="EPSG:3857")

        result = _resolve_geometry_selection(intersects=intersects, bbox=bbox, bbox_crs="EPSG:4326")

        np.testing.assert_array_equal(result.total_bounds, np.array([0.0, 0.0, 1.0, 1.0]))
        self.assertEqual(result.crs, "EPSG:4326")

    def test_intersects_used_when_bbox_missing(self):
        intersects = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)])

        result = _resolve_geometry_selection(intersects=intersects, bbox=None, bbox_crs="EPSG:4326")

        np.testing.assert_array_equal(result.total_bounds, np.array([0.0, 0.0, 2.0, 2.0]))
        self.assertEqual(result.crs, "EPSG:4326")


class TestApplyCloudMask(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        times = np.array(["2024-01-01", "2024-01-05", "2024-01-10"], dtype="datetime64[ns]")
        x = np.arange(10)
        y = np.arange(10)

        self.ds = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(3, 10, 10) * 1000),
                "nir": (["time", "y", "x"], np.random.rand(3, 10, 10) * 1000),
                "cloud-mask": (
                    ["time", "y", "x"],
                    np.array(
                        [
                            np.ones((10, 10)),
                            np.where(np.random.rand(10, 10) > 0.5, 1, 2),
                            np.where(np.random.rand(10, 10) > 0.7, 1, 3),
                        ]
                    ).astype(np.int8),
                ),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.ds.rio.write_crs("EPSG:4326", inplace=True)

    def test_mask_band_not_found(self):
        with self.assertRaises(DatacubeMaskingError) as context:
            apply_cloud_mask(self.ds, mask_band="non-existent-mask")
        self.assertIn("'non-existent-mask' not found", str(context.exception))

    def test_strategy_no_clouds(self):
        result = apply_cloud_mask(self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=False)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("red", result.data_vars)
        self.assertIn("nir", result.data_vars)
        self.assertNotIn("cloud-mask", result.data_vars)
        masked_pixels = result["red"].isel(time=1).isnull().sum().values
        self.assertGreater(masked_pixels, 0)

    def test_strategy_no_clouds_snow_ice(self):
        result = apply_cloud_mask(self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=False)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("red", result.data_vars)

    def test_strategy_no_clouds_snow_ice_water(self):
        result = apply_cloud_mask(self.ds, mask_band="cloud-mask", exclude_values=[2, 3, 4], mask_statistics=False)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("red", result.data_vars)

    def test_custom_mask_function(self):
        def custom_func(ds):
            return ds["cloud-mask"] == 1

        result = apply_cloud_mask(
            self.ds, mask_band="cloud-mask", custom_mask_function=custom_func, mask_statistics=False
        )
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("red", result.data_vars)

    def test_include_mask_band(self):
        result = apply_cloud_mask(
            self.ds, mask_band="cloud-mask", clear_values=[1], include_mask_band=True, mask_statistics=False
        )
        self.assertIn("cloud-mask", result.data_vars)

    def test_exclude_mask_band(self):
        result = apply_cloud_mask(
            self.ds, mask_band="cloud-mask", clear_values=[1], include_mask_band=False, mask_statistics=False
        )
        self.assertNotIn("cloud-mask", result.data_vars)

    def test_mask_statistics_computed(self):
        result = apply_cloud_mask(self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True)
        self.assertIn("clear_percent", result.coords)
        self.assertIn("clear_pixels", result.coords)
        self.assertEqual(len(result["clear_percent"]), 3)
        self.assertEqual(len(result["clear_pixels"]), 3)

    def test_mask_statistics_not_computed(self):
        result = apply_cloud_mask(self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=False)
        self.assertNotIn("clear_percent", result.coords)
        self.assertNotIn("clear_pixels", result.coords)

    def test_clear_percent_values(self):
        result = apply_cloud_mask(self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True)
        self.assertEqual(result["clear_percent"].isel(time=0).values, 100)
        self.assertLess(result["clear_percent"].isel(time=1).values, 100)

    def test_clear_pixels_values(self):
        result = apply_cloud_mask(self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True)
        self.assertEqual(result["clear_pixels"].isel(time=0).values, 100)
        self.assertLess(result["clear_pixels"].isel(time=1).values, 100)

    def test_fill_value_nan(self):
        result = apply_cloud_mask(
            self.ds, mask_band="cloud-mask", clear_values=[1], fill_value=np.nan, mask_statistics=False
        )
        self.assertTrue(np.isnan(result["red"].isel(time=1).values).any())

    def test_fill_value_zero(self):
        result = apply_cloud_mask(
            self.ds, mask_band="cloud-mask", clear_values=[1], fill_value=0, mask_statistics=False
        )
        masked_values = result["red"].isel(time=1).values
        self.assertTrue((masked_values[masked_values == 0].size > 0))

    def test_round_time(self):
        ds_with_ns = self.ds.copy()
        ds_with_ns["time"] = ds_with_ns["time"] + np.timedelta64(123456789, "ns")
        original_time_str = str(ds_with_ns["time"].values[0])
        self.assertIn("123456789", original_time_str)

        result = apply_cloud_mask(
            ds_with_ns, mask_band="cloud-mask", clear_values=[1], round_time=True, mask_statistics=False
        )
        rounded_time_str = str(result["time"].values[0])
        self.assertNotIn("123456789", rounded_time_str)
        self.assertTrue(rounded_time_str.endswith("000000000") or ":00" in rounded_time_str)

    def test_round_time_requires_datetime(self):
        ds_with_string_time = self.ds.copy()
        ds_with_string_time["time"] = ["2024-01-01", "2024-01-02", "2024-01-03"]
        with self.assertRaises(DatacubeMaskingError) as context:
            apply_cloud_mask(
                ds_with_string_time, mask_band="cloud-mask", clear_values=[1], round_time=True, mask_statistics=False
            )
        self.assertIn("round_time requires time coordinate to be datetime type", str(context.exception))

    def test_clear_cover_filtering(self):
        result = apply_cloud_mask(
            self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True, clear_cover=80
        )
        self.assertLessEqual(len(result.time), 3)
        for t in range(len(result.time)):
            self.assertGreaterEqual(result["clear_percent"].isel(time=t).values, 80)

    def test_clear_cover_requires_statistics(self):
        with self.assertRaises(DatacubeMaskingError) as context:
            apply_cloud_mask(self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=False, clear_cover=50)
        self.assertIn("clear_cover filtering requires mask_statistics=True", str(context.exception))

    def test_clear_cover_filtering_with_dask_arrays(self):
        """Test clear_cover filtering works with lazy dask arrays (common in real datacubes)"""
        times = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")

        # Create dataset with dask-backed arrays (lazy evaluation)
        ds_dask = xr.Dataset(
            {
                "red": (["time", "y", "x"], da.from_array(np.random.rand(3, 10, 10) * 1000, chunks=(1, 10, 10))),
                "green": (["time", "y", "x"], da.from_array(np.random.rand(3, 10, 10) * 1000, chunks=(1, 10, 10))),
                "cloud-mask": (
                    ["time", "y", "x"],
                    da.from_array(
                        np.array(
                            [
                                np.ones((10, 10), dtype=np.int8),  # 100% clear
                                np.where(np.arange(100).reshape(10, 10) < 50, 1, 2),  # 50% clear
                                np.full((10, 10), 2, dtype=np.int8),  # 0% clear
                            ]
                        ),
                        chunks=(1, 10, 10),
                    ),
                ),
            },
            coords={"time": times, "y": range(10), "x": range(10)},
        )
        ds_dask.rio.write_crs("EPSG:4326", inplace=True)

        # This should not crash with "boolean dask array not allowed" error
        result = apply_cloud_mask(
            ds_dask, mask_band="cloud-mask", clear_values=[1], mask_statistics=True, clear_cover=60
        )

        # Should filter out timesteps with <60% clear coverage
        self.assertEqual(len(result.time), 1)  # Only first timestep (100% clear) passes
        self.assertEqual(result["clear_percent"].values[0], 100)

    def test_usable_pixels_attribute(self):
        result = apply_cloud_mask(self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True)
        self.assertIn("usable_pixels", result.attrs)
        self.assertEqual(result.attrs["usable_pixels"], 100)

    def test_dimension_mismatch_raises_error(self):
        ds_multi = self.ds.copy()
        ds_multi["reduced_var"] = (["time", "y"], np.random.rand(3, 10))

        with self.assertRaises(DatacubeMaskingError) as context:
            apply_cloud_mask(ds_multi, mask_band="cloud-mask", clear_values=[1], mask_statistics=False)

        self.assertIn("Cannot mask variable", str(context.exception))
        self.assertIn("reduced_var", str(context.exception))
        self.assertIn("extra dimensions", str(context.exception))

    def test_data_preservation(self):
        result = apply_cloud_mask(self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=False)
        first_timestep_clear = result["red"].isel(time=0)
        original_first_timestep = self.ds["red"].isel(time=0)
        np.testing.assert_array_equal(first_timestep_clear.values, original_first_timestep.values)

    def test_multiple_strategies_produce_different_results(self):
        times = np.array(["2024-01-01"], dtype="datetime64[ns]")
        x = np.arange(10)
        y = np.arange(10)

        mask_data = np.ones((10, 10), dtype=np.int8)
        mask_data[0:3, :] = 2
        mask_data[3:6, :] = 3
        mask_data[6:8, :] = 4

        ds_test = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(1, 10, 10) * 1000),
                "cloud-mask": (["time", "y", "x"], mask_data[np.newaxis, :, :]),
            },
            coords={"time": times, "y": y, "x": x},
        )
        ds_test.rio.write_crs("EPSG:4326", inplace=True)

        result_clear_only = apply_cloud_mask(ds_test, clear_values=[1], mask_statistics=True)
        result_exclude_some = apply_cloud_mask(ds_test, exclude_values=[2, 3], mask_statistics=True)
        result_exclude_all = apply_cloud_mask(ds_test, exclude_values=[2, 3, 4], mask_statistics=True)

        self.assertEqual(result_clear_only["clear_percent"].values[0], 20)
        self.assertEqual(result_exclude_some["clear_percent"].values[0], 40)
        self.assertEqual(result_exclude_all["clear_percent"].values[0], 20)

    def test_default_behavior_uses_clear_value_one(self):
        ds_test = xr.Dataset(
            {
                "red": xr.DataArray(np.ones((1, 10, 10)), dims=("time", "y", "x")),
                "cloud-mask": xr.DataArray(
                    np.array([[[1, 1, 2, 2, 2, 2, 2, 2, 2, 2] for _ in range(10)]]), dims=("time", "y", "x")
                ),
            }
        )
        result = apply_cloud_mask(ds_test, clear_values=[1], mask_statistics=False)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("red", result.data_vars)
        self.assertFalse(np.isnan(result["red"].values[0, 0, 0]))
        self.assertTrue(np.isnan(result["red"].values[0, 0, 2]))

    def test_requires_masking_parameter(self):
        with self.assertRaises(DatacubeMaskingError) as context:
            apply_cloud_mask(self.ds, mask_band="cloud-mask", mask_statistics=False)
        self.assertIn("Must provide one of", str(context.exception))


class TestApplyCloudMaskWithGeometry(unittest.TestCase):
    def setUp(self):
        import geopandas as gpd
        from shapely.geometry import box

        np.random.seed(42)
        times = np.array(["2024-01-01", "2024-01-05"], dtype="datetime64[ns]")
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)

        self.ds = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(2, 20, 20) * 1000),
                "cloud-mask": (["time", "y", "x"], np.ones((2, 20, 20), dtype=np.int8)),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.ds.rio.write_crs("EPSG:4326", inplace=True)
        self.ds.rio.write_transform(inplace=True)

        self.intersects = gpd.GeoDataFrame(geometry=[box(0.25, 0.25, 0.75, 0.75)], crs="EPSG:4326")

    def test_geometry_aware_pixel_counting(self):
        result = apply_cloud_mask(
            self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True, intersects=self.intersects
        )
        self.assertIn("usable_pixels", result.attrs)
        self.assertLess(result.attrs["usable_pixels"], 400)
        self.assertGreater(result.attrs["usable_pixels"], 0)

    def test_bbox_conversion(self):
        bbox = [0.25, 0.25, 0.75, 0.75]
        result = apply_cloud_mask(self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True, bbox=bbox)
        self.assertIn("usable_pixels", result.attrs)
        self.assertLess(result.attrs["usable_pixels"], 400)

    def test_intersects_as_dict(self):
        intersects_dict = {
            "type": "Polygon",
            "coordinates": [[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.25, 0.25]]],
        }
        result = apply_cloud_mask(
            self.ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True, intersects=intersects_dict
        )
        self.assertIn("usable_pixels", result.attrs)
        self.assertLess(result.attrs["usable_pixels"], 400)

    def test_clear_percent_never_exceeds_100_with_bbox(self):
        times = np.array(["2024-01-01", "2024-01-05"], dtype="datetime64[ns]")
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)

        cloud_mask = np.ones((2, 50, 50), dtype=np.int8)
        cloud_mask[:, 40:, 40:] = 2

        ds = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(2, 50, 50) * 1000),
                "cloud-mask": (["time", "y", "x"], cloud_mask),
            },
            coords={"time": times, "y": y, "x": x},
        )
        ds.rio.write_crs("EPSG:4326", inplace=True)
        ds.rio.write_transform(inplace=True)

        bbox = [0.1, 0.1, 0.3, 0.3]
        result = apply_cloud_mask(ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True, bbox=bbox)

        self.assertIn("clear_percent", result.coords)
        for t in range(len(result.time)):
            clear_pct = float(result["clear_percent"].isel(time=t).values)
            self.assertLessEqual(clear_pct, 100.0, f"clear_percent exceeds 100% at time {t}: {clear_pct}")
            self.assertGreaterEqual(clear_pct, 0.0, f"clear_percent is negative at time {t}: {clear_pct}")

    def test_geometry_clip_uses_dynamic_spatial_dims(self):
        times = np.array(["2024-01-01", "2024-01-05"], dtype="datetime64[ns]")
        y = np.linspace(0, 1, 30)
        x = np.linspace(0, 1, 30)

        cloud_mask = np.ones((2, 30, 30), dtype=np.int8)
        cloud_mask[:, 20:, 20:] = 2

        ds = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(2, 30, 30) * 1000),
                "cloud-mask": (["time", "y", "x"], cloud_mask),
            },
            coords={"time": times, "y": y, "x": x},
        )
        ds.rio.write_crs("EPSG:4326", inplace=True)
        ds.rio.write_transform(inplace=True)

        bbox = [0.1, 0.1, 0.6, 0.6]
        result = apply_cloud_mask(ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True, bbox=bbox)

        self.assertIn("clear_percent", result.coords)
        self.assertIn("usable_pixels", result.attrs)
        usable = result.attrs["usable_pixels"]
        self.assertGreater(usable, 0)
        self.assertLess(usable, 900)

        for t in range(len(result.time)):
            clear_pct = float(result["clear_percent"].isel(time=t).values)
            self.assertLessEqual(clear_pct, 100.0)
            self.assertGreaterEqual(clear_pct, 0.0)


class TestApplyCloudMaskWithSeparateDataset(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        times = np.array(["2024-01-01", "2024-01-05", "2024-01-10"], dtype="datetime64[ns]")
        x = np.arange(10)
        y = np.arange(10)

        self.ds_data = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(3, 10, 10) * 1000),
                "nir": (["time", "y", "x"], np.random.rand(3, 10, 10) * 1000),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.ds_data.rio.write_crs("EPSG:4326", inplace=True)

        self.ds_mask = xr.Dataset(
            {
                "cloud-mask": (
                    ["time", "y", "x"],
                    np.array(
                        [
                            np.ones((10, 10)),
                            np.where(np.random.rand(10, 10) > 0.5, 1, 2),
                            np.where(np.random.rand(10, 10) > 0.7, 1, 3),
                        ]
                    ).astype(np.int8),
                ),
            },
            coords={"time": times, "y": y, "x": x},
        )
        self.ds_mask.rio.write_crs("EPSG:4326", inplace=True)

        self.ds_combined = xr.merge([self.ds_data, self.ds_mask], compat="override")
        self.ds_combined.rio.write_crs("EPSG:4326", inplace=True)

    def test_mask_dataset_basic_merge(self):
        result = apply_cloud_mask(
            self.ds_data, mask_dataset=self.ds_mask, clear_values=[1], mask_band="cloud-mask", mask_statistics=False
        )
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("red", result.data_vars)
        self.assertIn("nir", result.data_vars)
        self.assertNotIn("cloud-mask", result.data_vars)

    def test_mask_dataset_equivalence_to_merged(self):
        result_separate = apply_cloud_mask(
            self.ds_data, mask_dataset=self.ds_mask, clear_values=[1], mask_band="cloud-mask", mask_statistics=False
        )
        result_merged = apply_cloud_mask(
            self.ds_combined, clear_values=[1], mask_band="cloud-mask", mask_statistics=False
        )

        np.testing.assert_array_equal(result_separate["red"].values, result_merged["red"].values)
        np.testing.assert_array_equal(result_separate["nir"].values, result_merged["nir"].values)

    def test_mask_dataset_statistics_equivalence(self):
        result_separate = apply_cloud_mask(
            self.ds_data, mask_dataset=self.ds_mask, clear_values=[1], mask_band="cloud-mask", mask_statistics=True
        )
        result_merged = apply_cloud_mask(
            self.ds_combined, clear_values=[1], mask_band="cloud-mask", mask_statistics=True
        )

        np.testing.assert_array_equal(result_separate["clear_percent"].values, result_merged["clear_percent"].values)
        np.testing.assert_array_equal(result_separate["clear_pixels"].values, result_merged["clear_pixels"].values)

    def test_mask_dataset_with_bbox(self):
        bbox = [0, 0, 5, 5]
        result = apply_cloud_mask(
            self.ds_data,
            mask_dataset=self.ds_mask,
            clear_values=[1],
            mask_band="cloud-mask",
            mask_statistics=True,
            bbox=bbox,
        )
        self.assertIn("usable_pixels", result.attrs)
        self.assertIn("clear_percent", result.coords)

    def test_mask_dataset_with_clear_cover_filtering(self):
        result = apply_cloud_mask(
            self.ds_data,
            mask_dataset=self.ds_mask,
            clear_values=[1],
            mask_band="cloud-mask",
            mask_statistics=True,
            clear_cover=80,
        )
        self.assertLessEqual(len(result.time), 3)
        for t in range(len(result.time)):
            self.assertGreaterEqual(result["clear_percent"].isel(time=t).values, 80)

    def test_mask_dataset_include_mask_band(self):
        result = apply_cloud_mask(
            self.ds_data,
            mask_dataset=self.ds_mask,
            clear_values=[1],
            mask_band="cloud-mask",
            include_mask_band=True,
            mask_statistics=False,
        )
        self.assertIn("cloud-mask", result.data_vars)

    def test_mask_dataset_with_fill_value(self):
        result = apply_cloud_mask(
            self.ds_data,
            mask_dataset=self.ds_mask,
            clear_values=[1],
            mask_band="cloud-mask",
            fill_value=-9999,
            mask_statistics=False,
        )
        masked_pixels = result["red"].isel(time=1).values
        self.assertTrue((masked_pixels == -9999).any())

    def test_mask_dataset_none_uses_existing_mask(self):
        result = apply_cloud_mask(
            self.ds_combined, mask_dataset=None, clear_values=[1], mask_band="cloud-mask", mask_statistics=False
        )
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("red", result.data_vars)


class TestApplyCloudMaskEdgeCases(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_single_timestep(self):
        times = np.array(["2024-01-01"], dtype="datetime64[ns]")
        x = np.arange(5)
        y = np.arange(5)

        ds = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(1, 5, 5) * 1000),
                "cloud-mask": (["time", "y", "x"], np.ones((1, 5, 5), dtype=np.int8)),
            },
            coords={"time": times, "y": y, "x": x},
        )
        ds.rio.write_crs("EPSG:4326", inplace=True)

        result = apply_cloud_mask(ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True)
        self.assertEqual(len(result.time), 1)
        self.assertEqual(result["clear_percent"].values[0], 100)

    def test_all_masked(self):
        times = np.array(["2024-01-01", "2024-01-05"], dtype="datetime64[ns]")
        x = np.arange(5)
        y = np.arange(5)

        ds = xr.Dataset(
            {
                "red": (["time", "y", "x"], np.random.rand(2, 5, 5) * 1000),
                "cloud-mask": (["time", "y", "x"], np.full((2, 5, 5), 2, dtype=np.int8)),
            },
            coords={"time": times, "y": y, "x": x},
        )
        ds.rio.write_crs("EPSG:4326", inplace=True)

        result = apply_cloud_mask(ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True)
        self.assertEqual(result["clear_percent"].values[0], 0)
        self.assertTrue(result["red"].isel(time=0).isnull().all())

    def test_no_time_dimension(self):
        x = np.arange(5)
        y = np.arange(5)

        ds = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(5, 5) * 1000),
                "cloud-mask": (["y", "x"], np.ones((5, 5), dtype=np.int8)),
            },
            coords={"y": y, "x": x},
        )
        ds.rio.write_crs("EPSG:4326", inplace=True)

        result = apply_cloud_mask(ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=False)
        self.assertIn("red", result.data_vars)

    def test_non_xy_spatial_dimensions(self):
        times = np.array(["2024-01-01"], dtype="datetime64[ns]")
        lat = np.arange(5)
        lon = np.arange(5)

        ds = xr.Dataset(
            {
                "red": (["time", "lat", "lon"], np.random.rand(1, 5, 5) * 1000),
                "cloud-mask": (["time", "lat", "lon"], np.ones((1, 5, 5), dtype=np.int8)),
            },
            coords={"time": times, "lat": lat, "lon": lon},
        )
        ds.rio.write_crs("EPSG:4326", inplace=True)

        result = apply_cloud_mask(ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True)
        self.assertIn("clear_percent", result.coords)
        self.assertEqual(result["clear_percent"].values[0], 100)

    def test_clear_cover_filtering_without_time_dimension_pass(self):
        """Test clear_cover filtering on dataset without time dimension that passes threshold"""
        x = np.arange(5)
        y = np.arange(5)

        ds = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(5, 5) * 1000),
                "cloud-mask": (["y", "x"], np.ones((5, 5), dtype=np.int8)),
            },
            coords={"y": y, "x": x},
        )
        ds.rio.write_crs("EPSG:4326", inplace=True)

        # 100% clear, so threshold of 80% should pass
        result = apply_cloud_mask(ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True, clear_cover=80)
        self.assertIn("red", result.data_vars)
        self.assertEqual(result["clear_percent"].values, 100)

    def test_clear_cover_filtering_without_time_dimension_fail(self):
        """Test clear_cover filtering on dataset without time dimension that fails threshold"""
        x = np.arange(5)
        y = np.arange(5)

        ds = xr.Dataset(
            {
                "red": (["y", "x"], np.random.rand(5, 5) * 1000),
                "cloud-mask": (["y", "x"], np.zeros((5, 5), dtype=np.int8)),  # All masked
            },
            coords={"y": y, "x": x},
        )
        ds.rio.write_crs("EPSG:4326", inplace=True)

        # 0% clear, so threshold of 50% should fail
        with self.assertRaises(DatacubeMaskingError) as context:
            apply_cloud_mask(ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True, clear_cover=50)
        self.assertIn("does not meet minimum clear_cover threshold", str(context.exception))

    def test_clear_cover_filtering_with_multidimensional_clear_percent(self):
        """Test that clear_cover filtering raises error for multi-dimensional clear_percent"""
        times = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
        x = np.arange(5)
        y = np.arange(5)
        z = np.arange(3)

        # Create a dataset with 4 dimensions (time, z, y, x)
        ds = xr.Dataset(
            {
                "red": (["time", "z", "y", "x"], np.random.rand(2, 3, 5, 5) * 1000),
                "cloud-mask": (["time", "z", "y", "x"], np.ones((2, 3, 5, 5), dtype=np.int8)),
            },
            coords={"time": times, "z": z, "y": y, "x": x},
        )
        ds.rio.write_crs("EPSG:4326", inplace=True)

        # This will produce clear_percent with dims (time, z) after summing over x,y
        with self.assertRaises(DatacubeMaskingError) as context:
            apply_cloud_mask(ds, mask_band="cloud-mask", clear_values=[1], mask_statistics=True, clear_cover=50)
        self.assertIn("clear_cover filtering not supported for multi-dimensional clear_percent", str(context.exception))


class TestApplyCloudMaskCustomPixelValues(unittest.TestCase):
    def test_custom_clear_values(self):
        ds = xr.Dataset(
            {
                "blue": xr.DataArray(np.ones((1, 10, 10)), dims=("time", "y", "x")),
                "cloud-mask": xr.DataArray(
                    np.array([[[5, 5, 5, 5, 5, 5, 5, 5, 5, 5] for _ in range(10)]]), dims=("time", "y", "x")
                ),
            }
        )
        result = apply_cloud_mask(ds, clear_values=[5], mask_statistics=False)
        np.testing.assert_array_equal(result["blue"].values, ds["blue"].values)

    def test_exclude_values_single(self):
        ds = xr.Dataset(
            {
                "blue": xr.DataArray(np.ones((1, 10, 10)), dims=("time", "y", "x")),
                "cloud-mask": xr.DataArray(
                    np.array([[[10, 10, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(10)]]), dims=("time", "y", "x")
                ),
            }
        )
        result = apply_cloud_mask(ds, exclude_values=[10], mask_statistics=False)
        self.assertTrue(np.isnan(result["blue"].values[0, 0, 0]))
        self.assertFalse(np.isnan(result["blue"].values[0, 0, 2]))

    def test_exclude_values_multiple(self):
        ds = xr.Dataset(
            {
                "blue": xr.DataArray(np.ones((1, 10, 10)), dims=("time", "y", "x")),
                "cloud-mask": xr.DataArray(
                    np.array([[[10, 11, 12, 1, 1, 1, 1, 1, 1, 1] for _ in range(10)]]), dims=("time", "y", "x")
                ),
            }
        )
        result = apply_cloud_mask(ds, exclude_values=[10, 11, 12], mask_statistics=False)
        self.assertTrue(np.isnan(result["blue"].values[0, 0, 0]))
        self.assertTrue(np.isnan(result["blue"].values[0, 0, 1]))
        self.assertTrue(np.isnan(result["blue"].values[0, 0, 2]))
        self.assertFalse(np.isnan(result["blue"].values[0, 0, 3]))

    def test_clear_values_multiple(self):
        ds = xr.Dataset(
            {
                "blue": xr.DataArray(np.ones((1, 10, 10)), dims=("time", "y", "x")),
                "cloud-mask": xr.DataArray(
                    np.array([[[1, 5, 7, 2, 2, 2, 2, 2, 2, 2] for _ in range(10)]]), dims=("time", "y", "x")
                ),
            }
        )
        result = apply_cloud_mask(ds, clear_values=[1, 5, 7], mask_statistics=False)
        self.assertFalse(np.isnan(result["blue"].values[0, 0, 0]))
        self.assertFalse(np.isnan(result["blue"].values[0, 0, 1]))
        self.assertFalse(np.isnan(result["blue"].values[0, 0, 2]))
        self.assertTrue(np.isnan(result["blue"].values[0, 0, 3]))

    def test_clear_values_takes_precedence(self):
        ds = xr.Dataset(
            {
                "blue": xr.DataArray(np.ones((1, 10, 10)), dims=("time", "y", "x")),
                "cloud-mask": xr.DataArray(
                    np.array([[[1, 2, 3, 4, 5, 5, 5, 5, 5, 5] for _ in range(10)]]), dims=("time", "y", "x")
                ),
            }
        )
        result = apply_cloud_mask(ds, clear_values=[1, 2], exclude_values=[3, 4], mask_statistics=False)
        self.assertFalse(np.isnan(result["blue"].values[0, 0, 0]))
        self.assertFalse(np.isnan(result["blue"].values[0, 0, 1]))
        self.assertTrue(np.isnan(result["blue"].values[0, 0, 2]))
        self.assertTrue(np.isnan(result["blue"].values[0, 0, 3]))

    def test_eda_clear_values_behavior(self):
        ds = xr.Dataset(
            {
                "blue": xr.DataArray(np.ones((1, 10, 10)), dims=("time", "y", "x")),
                "cloud-mask": xr.DataArray(
                    np.array([[[2, 3, 4, 1, 1, 1, 1, 1, 1, 1] for _ in range(10)]]), dims=("time", "y", "x")
                ),
            }
        )
        result = apply_cloud_mask(ds, clear_values=[1], mask_statistics=False)
        self.assertTrue(np.isnan(result["blue"].values[0, 0, 0]))
        self.assertTrue(np.isnan(result["blue"].values[0, 0, 1]))
        self.assertTrue(np.isnan(result["blue"].values[0, 0, 2]))
        self.assertFalse(np.isnan(result["blue"].values[0, 0, 3]))


if __name__ == "__main__":
    unittest.main()
