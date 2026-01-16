import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr
from pystac import Item

from earthdaily.datacube._datacube import Datacube
from earthdaily.datacube._datacube_service import DatacubeService
from earthdaily.datacube.constants import DEFAULT_DTYPE, DEFAULT_ENGINE, DEFAULT_HREF_PATH, DEFAULT_NODATA


class TestDatacubeService(unittest.TestCase):
    def setUp(self):
        self.service = DatacubeService()
        self.mock_item = MagicMock(spec=Item)
        self.mock_items = [self.mock_item]

        times = pd.date_range("2024-01-01", periods=3, freq="D")
        x = np.arange(10)
        y = np.arange(10)
        self.mock_dataset = xr.Dataset(
            {
                "band1": (["time", "y", "x"], np.random.rand(3, 10, 10)),
                "band2": (["time", "y", "x"], np.random.rand(3, 10, 10)),
            },
            coords={"time": times, "y": y, "x": x},
        )

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_defaults(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        result = self.service.create(items=self.mock_items)

        self.assertIsInstance(result, Datacube)
        mock_build_datacube.assert_called_once_with(
            items=self.mock_items,
            assets=None,
            bbox=None,
            intersects=None,
            dtype=DEFAULT_DTYPE,
            nodata=DEFAULT_NODATA,
            properties=False,
            engine=DEFAULT_ENGINE,
            replace_href_with=DEFAULT_HREF_PATH,
        )

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_custom_parameters(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        custom_assets = ["red", "green", "blue"]
        custom_bbox = [0, 0, 10, 10]
        custom_dtype = "float64"
        custom_nodata = -9999.0
        custom_engine = "odc"
        custom_href_path = "custom.href.path"

        result = self.service.create(
            items=self.mock_items,
            assets=custom_assets,
            bbox=custom_bbox,
            dtype=custom_dtype,
            nodata=custom_nodata,
            engine=custom_engine,
            replace_href_with=custom_href_path,
        )

        self.assertIsInstance(result, Datacube)
        mock_build_datacube.assert_called_once_with(
            items=self.mock_items,
            assets=custom_assets,
            bbox=custom_bbox,
            intersects=None,
            dtype=custom_dtype,
            nodata=custom_nodata,
            properties=False,
            engine=custom_engine,
            replace_href_with=custom_href_path,
        )

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_intersects(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        mock_intersects = MagicMock()

        result = self.service.create(items=self.mock_items, intersects=mock_intersects)

        self.assertIsInstance(result, Datacube)
        mock_build_datacube.assert_called_once_with(
            items=self.mock_items,
            assets=None,
            bbox=None,
            intersects=mock_intersects,
            dtype=DEFAULT_DTYPE,
            nodata=DEFAULT_NODATA,
            properties=False,
            engine=DEFAULT_ENGINE,
            replace_href_with=DEFAULT_HREF_PATH,
        )

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_properties(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        result = self.service.create(items=self.mock_items, properties=True)

        self.assertIsInstance(result, Datacube)
        mock_build_datacube.assert_called_once_with(
            items=self.mock_items,
            assets=None,
            bbox=None,
            intersects=None,
            dtype=DEFAULT_DTYPE,
            nodata=DEFAULT_NODATA,
            properties=True,
            engine=DEFAULT_ENGINE,
            replace_href_with=DEFAULT_HREF_PATH,
        )

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_properties_list(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        properties_list = ["eo:cloud_cover", "datetime"]

        result = self.service.create(items=self.mock_items, properties=properties_list)

        self.assertIsInstance(result, Datacube)
        mock_build_datacube.assert_called_once_with(
            items=self.mock_items,
            assets=None,
            bbox=None,
            intersects=None,
            dtype=DEFAULT_DTYPE,
            nodata=DEFAULT_NODATA,
            properties=properties_list,
            engine=DEFAULT_ENGINE,
            replace_href_with=DEFAULT_HREF_PATH,
        )

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_kwargs(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        extra_kwargs = {"chunks": {"x": 100, "y": 100}, "resampling": "bilinear"}

        result = self.service.create(items=self.mock_items, **extra_kwargs)

        self.assertIsInstance(result, Datacube)
        call_kwargs = mock_build_datacube.call_args[1]
        self.assertIn("chunks", call_kwargs)
        self.assertIn("resampling", call_kwargs)

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_returns_datacube_instance(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        result = self.service.create(items=self.mock_items)

        self.assertIsInstance(result, Datacube)
        self.assertEqual(result._metadata["items_count"], len(self.mock_items))
        self.assertEqual(result._metadata["engine"], DEFAULT_ENGINE)

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_metadata_contains_items_count(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        result = self.service.create(items=self.mock_items)

        self.assertIsInstance(result, Datacube)
        self.assertEqual(result._metadata["items_count"], len(self.mock_items))
        self.assertEqual(result._metadata["engine"], DEFAULT_ENGINE)

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_empty_items_list(self, mock_build_datacube):
        empty_dataset = xr.Dataset()
        mock_build_datacube.return_value = empty_dataset

        result = self.service.create(items=[])

        self.assertIsInstance(result, Datacube)
        mock_build_datacube.assert_called_once_with(
            items=[],
            assets=None,
            bbox=None,
            intersects=None,
            dtype=DEFAULT_DTYPE,
            nodata=DEFAULT_NODATA,
            properties=False,
            engine=DEFAULT_ENGINE,
            replace_href_with=DEFAULT_HREF_PATH,
        )

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_multiple_items(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        multiple_items = [MagicMock(spec=Item), MagicMock(spec=Item), MagicMock(spec=Item)]

        result = self.service.create(items=multiple_items)

        self.assertIsInstance(result, Datacube)
        mock_build_datacube.assert_called_once()
        call_kwargs = mock_build_datacube.call_args[1]
        self.assertEqual(len(call_kwargs["items"]), 3)

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_assets_dict(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        assets_dict = {"red": "B04", "green": "B03", "blue": "B02"}

        result = self.service.create(items=self.mock_items, assets=assets_dict)

        self.assertIsInstance(result, Datacube)
        mock_build_datacube.assert_called_once_with(
            items=self.mock_items,
            assets=assets_dict,
            bbox=None,
            intersects=None,
            dtype=DEFAULT_DTYPE,
            nodata=DEFAULT_NODATA,
            properties=False,
            engine=DEFAULT_ENGINE,
            replace_href_with=DEFAULT_HREF_PATH,
        )

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_bbox_tuple(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        bbox_tuple = (0.0, 0.0, 10.0, 10.0)

        result = self.service.create(items=self.mock_items, bbox=bbox_tuple)

        self.assertIsInstance(result, Datacube)
        mock_build_datacube.assert_called_once_with(
            items=self.mock_items,
            assets=None,
            bbox=bbox_tuple,
            intersects=None,
            dtype=DEFAULT_DTYPE,
            nodata=DEFAULT_NODATA,
            properties=False,
            engine=DEFAULT_ENGINE,
            replace_href_with=DEFAULT_HREF_PATH,
        )

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    def test_create_with_none_nodata(self, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        result = self.service.create(items=self.mock_items, nodata=None)

        self.assertIsInstance(result, Datacube)
        mock_build_datacube.assert_called_once_with(
            items=self.mock_items,
            assets=None,
            bbox=None,
            intersects=None,
            dtype=DEFAULT_DTYPE,
            nodata=None,
            properties=False,
            engine=DEFAULT_ENGINE,
            replace_href_with=DEFAULT_HREF_PATH,
        )

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    @patch("earthdaily.datacube._datacube_service._deduplicate_items")
    def test_create_with_deduplicate_by(self, mock_deduplicate, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset
        mock_deduplicate.return_value = self.mock_items

        result = self.service.create(
            items=self.mock_items,
            deduplicate_by=["date", "proj:transform"],
            deduplicate_keep="last",
        )

        self.assertIsInstance(result, Datacube)
        mock_deduplicate.assert_called_once_with(self.mock_items, ["date", "proj:transform"], "last")

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    @patch("earthdaily.datacube._datacube_service._deduplicate_items")
    def test_create_without_deduplicate_by_skips_deduplication(self, mock_deduplicate, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset

        result = self.service.create(items=self.mock_items)

        self.assertIsInstance(result, Datacube)
        mock_deduplicate.assert_not_called()

    @patch("earthdaily.datacube._datacube_service.build_datacube")
    @patch("earthdaily.datacube._datacube_service._deduplicate_items")
    def test_create_metadata_reflects_deduplicated_count(self, mock_deduplicate, mock_build_datacube):
        mock_build_datacube.return_value = self.mock_dataset
        deduplicated_items = [MagicMock(spec=Item), MagicMock(spec=Item)]
        mock_deduplicate.return_value = deduplicated_items

        original_items = [MagicMock(spec=Item) for _ in range(5)]
        result = self.service.create(
            items=original_items,
            deduplicate_by=["date"],
        )

        self.assertEqual(result._metadata["items_count"], 2)


if __name__ == "__main__":
    unittest.main()
