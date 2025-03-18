import unittest
from datetime import datetime

import numpy as np
import xarray as xr
from pystac import Item

from earthdaily.earthdatastore.cube_utils.harmonizer import Harmonizer


class TestHarmonizer(unittest.TestCase):
    def setUp(self):
        times = ["1987-04-22", "1998-07-29"]

        x_values = np.arange(0, 4)
        y_values = np.arange(0, 3)

        data_values = np.arange(0, 12).reshape(3, 4)
        data_values = np.dstack((data_values, data_values))

        self.fake_ds = xr.Dataset(
            {
                "first_var": (("y", "x", "time"), data_values),
            },
            coords={"y": y_values, "x": x_values, "time": times},
        )

    def generate_fake_xcal_item(self, id, published, expires, bands, source_platform):
        props = {
            "published": published,
            "eda_cross_cal:bands": bands,
            "eda_cross_cal:source_platform": source_platform,
        }

        if expires != "":
            props["expires"] = expires

        return Item(
            id=id,
            properties=props,
            bbox=[-180.0, -90.0, 180.0, 90.0],
            datetime=datetime.now,
            geometry={
                "coordinates": [
                    [
                        [-180.0, 90.0],
                        [-180.0, -90.0],
                        [180.0, -90.0],
                        [180.0, 90.0],
                        [-180.0, 90.0],
                    ]
                ],
                "type": "Polygon",
            },
        )

    def generate_fake_collection_item(self, id, datetime):
        return Item(
            id=id,
            bbox=[-180.0, -90.0, 180.0, 90.0],
            datetime=datetime,
            properties={"platform": ""},
            geometry={
                "coordinates": [
                    [
                        [-180.0, 90.0],
                        [-180.0, -90.0],
                        [180.0, -90.0],
                        [180.0, 90.0],
                        [-180.0, 90.0],
                    ]
                ],
                "type": "Polygon",
            },
        )

    def test_check_timerange(self):
        xcal_item_from_20230901_to_20230915 = self.generate_fake_xcal_item(
            "fake_xcal_item", "2023-09-01T00:00:00Z", "2023-09-15T23:59:59Z", {}, ""
        )

        valid_item_date = datetime.strptime(
            "2023-09-12T15:10:07Z", "%Y-%m-%dT%H:%M:%SZ"
        )

        invalid_item_date = datetime.strptime(
            "2024-09-27T05:40:02Z", "%Y-%m-%dT%H:%M:%SZ"
        )

        self.assertTrue(
            Harmonizer.check_timerange(
                xcal_item_from_20230901_to_20230915, valid_item_date
            )
        )

        self.assertFalse(
            Harmonizer.check_timerange(
                xcal_item_from_20230901_to_20230915, invalid_item_date
            )
        )

    def test_apply_to_asset_single_function(self):
        single_function_xcal = {
            "single_func": [{"single_func": [{"scale": 2, "offset": 0.5}]}]
        }

        fake_ds = self.fake_ds

        scaled_dataset = {}
        scaled_dataset["single_func"] = []

        for idx, time in enumerate(fake_ds.time.values):
            scaled_asset = Harmonizer.apply_to_asset(
                single_function_xcal["single_func"][0]["single_func"],
                fake_ds[["first_var"]].loc[dict(time=time)],
                "single_func",
            )
            scaled_dataset["single_func"].append(scaled_asset)

        ds_ = []
        for k, v in scaled_dataset.items():
            ds_k = []
            for d in v:
                ds_k.append(d)
            ds_.append(xr.concat(ds_k, dim="time"))
        ds_ = xr.merge(ds_).sortby("time")
        ds_.attrs = fake_ds.attrs

        expected_array = [
            [0.5, 2.5, 4.5, 6.5],
            [8.5, 10.5, 12.5, 14.5],
            [16.5, 18.5, 20.5, 22.5],
        ]

        try:
            np.testing.assert_array_equal(
                ds_["first_var"].loc[dict(time="1987-04-22")], expected_array
            )
        except AssertionError:
            self.assertTrue(False, "Single Function xcal is not applied correctly")

    def test_apply_to_asset_multiple_functions(self):
        multiple_function_xcal = {
            "first_var": [
                {
                    "first_var": [
                        {
                            "scale": 1,
                            "offset": 0.5,
                            "range_start": {"ge": 0},
                            "range_end": {"le": 6},
                        },
                        {
                            "scale": 2,
                            "offset": 0,
                            "range_start": {"gt": 6},
                            "range_end": {"le": 12},
                        },
                    ]
                }
            ]
        }

        # Creating simple dataset
        fake_ds = self.fake_ds

        scaled_dataset = {}
        scaled_dataset["first_var"] = []

        for idx, time in enumerate(fake_ds.time.values):
            scaled_asset = Harmonizer.apply_to_asset(
                multiple_function_xcal["first_var"][0]["first_var"],
                fake_ds["first_var"].loc[dict(time=time)],
                "first_var",
            )
            scaled_dataset["first_var"].append(scaled_asset)

        ds_ = []
        for k, v in scaled_dataset.items():
            ds_k = []
            for d in v:
                ds_k.append(d)
            ds_.append(xr.concat(ds_k, dim="time"))
        ds_ = xr.merge(ds_).sortby("time")
        ds_.attrs = fake_ds.attrs

        expected_array = [
            [0.5, 1.5, 2.5, 3.5],
            [4.5, 5.5, 6.5, 14],
            [16, 18, 20, 22],
        ]

        try:
            np.testing.assert_array_equal(
                ds_["first_var"].loc[dict(time="1987-04-22")], expected_array
            )
        except AssertionError:
            self.assertTrue(False, "Single Function xcal is not applied correctly")

    def test_different_xcal_by_date(self):
        fake_ds = self.fake_ds

        # Creating fake xcal items
        xcal_item_from_1980_to_1990 = self.generate_fake_xcal_item(
            "1980_1990",
            "1980-01-01T00:00:00Z",
            "1989-12-31T23:59:59Z",
            {"first_var": [{"first_var": [{"scale": 2, "offset": 10}]}]},
            "",
        )

        xcal_item_from_1990 = self.generate_fake_xcal_item(
            "1990_and_beyond",
            "1990-01-01T00:00:00Z",
            "",
            {"first_var": [{"first_var": [{"scale": 1, "offset": 0.5}]}]},
            "",
        )

        fake_xcal_item = [xcal_item_from_1980_to_1990, xcal_item_from_1990]

        # Creating fake item
        item_1987 = self.generate_fake_collection_item(
            "item_1987", "1987-04-22T12:18:59Z"
        )

        item_1998 = self.generate_fake_collection_item(
            "item_1998", "1998-07-29T12:18:59Z"
        )

        fake_items = [item_1987, item_1998]

        harmonized_fake_ds = Harmonizer.harmonize(
            fake_items, fake_ds, fake_xcal_item, ["first_var"]
        )

        expected_array_1987 = [[10, 12, 14, 16], [18, 20, 22, 24], [26, 28, 30, 32]]
        expected_array_1998 = [
            [0.5, 1.5, 2.5, 3.5],
            [4.5, 5.5, 6.5, 7.5],
            [8.5, 9.5, 10.5, 11.5],
        ]

        try:
            np.testing.assert_array_equal(
                harmonized_fake_ds["first_var"].loc[dict(time="1987-04-22")],
                expected_array_1987,
            )
        except AssertionError:
            self.assertTrue(False, "Incorrect xcal application for 1987-04-22")

        try:
            np.testing.assert_array_equal(
                harmonized_fake_ds["first_var"].loc[dict(time="1998-07-29")],
                expected_array_1998,
            )
        except AssertionError:
            self.assertTrue(False, "Incorrect xcal application for 1998-07-29")


if __name__ == "__main__":
    unittest.main()
