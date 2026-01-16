from __future__ import annotations

import unittest
from datetime import datetime
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr
from pystac import Asset, Item
from rasterio.enums import Resampling

from earthdaily.datacube._builder import (
    _deduplicate_items,
    _load_datacube_with_odc,
    _normalize_longitude_coordinate,
    _replace_item_hrefs,
    build_datacube,
    register_engine_loader,
    unregister_engine_loader,
)
from earthdaily.datacube.constants import (
    DEFAULT_BBOX_CRS,
    DIM_LATITUDE,
    DIM_LONGITUDE,
    DIM_TIME,
    DIM_X,
    DIM_Y,
)
from earthdaily.datacube.exceptions import DatacubeCreationError


class FakeAsset(Asset):
    def __init__(self, href: str = "original", extra_fields: dict[str, Any] | None = None) -> None:
        super().__init__(href=href, media_type=None, roles=None, title=None)
        self.extra_fields = extra_fields or {}


def make_fake_item(
    assets: dict[str, FakeAsset],
    properties: dict[str, Any] | None = None,
    dt: datetime | pd.Timestamp | None = None,
) -> Item:
    item = Item(
        id="fake-item",
        geometry=None,
        bbox=None,
        datetime=dt or datetime(2020, 1, 1, tzinfo=None),
        properties=properties or {},
    )
    for key, asset in assets.items():
        item.add_asset(key, asset)
    return item


class TestDatacubeBuilderEngines(unittest.TestCase):
    def setUp(self) -> None:
        self.time_index = pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"])

    def _make_dataset(self) -> xr.Dataset:
        data = xr.DataArray(
            np.zeros((2, 1, 2)),
            dims=(DIM_TIME, DIM_LATITUDE, DIM_LONGITUDE),
            coords={
                DIM_TIME: self.time_index,
                DIM_LATITUDE: ("latitude", np.array([10])),
                DIM_LONGITUDE: ("longitude", np.array([0, 270])),
            },
        )
        return xr.Dataset({"band": data})

    def _make_items(self) -> list[Item]:
        properties = [{"quality": 0.1}, {"quality": 0.2}, {"quality": 0.3}]
        datetimes = [
            pd.Timestamp("2020-01-01T00:00:00Z"),
            pd.Timestamp("2020-01-02T00:00:00Z"),
            None,
        ]
        items: list[Item] = []
        for idx in range(3):
            asset = FakeAsset(extra_fields={"alternate": {"download": {"href": f"alternate-{idx}.tif"}}})
            item = make_fake_item({"band": asset}, properties[idx], datetimes[idx])
            item.id = f"item-{idx}"
            items.append(item)
        return items

    def test_unknown_engine_error(self) -> None:
        with self.assertRaises(DatacubeCreationError) as ctx:
            build_datacube(items=[], engine="custom-engine")
        self.assertIn("custom-engine", str(ctx.exception))
        self.assertIn("Available engines", str(ctx.exception))

    def test_custom_engine_loader_invoked(self) -> None:
        engine_name = "dummy-engine"
        expected = xr.Dataset()

        def dummy_loader(**kwargs: Any) -> xr.Dataset:
            self.assertIn("items", kwargs)
            return expected

        register_engine_loader(engine_name, dummy_loader)
        try:
            result = build_datacube(items=[], engine=engine_name)
            self.assertIs(result, expected)
        finally:
            unregister_engine_loader(engine_name)

    def test_custom_loader_receives_nodata(self) -> None:
        engine_name = "nodata-engine"
        expected_nodata = -9999.0
        received: dict[str, Any] = {}

        def dummy_loader(**kwargs: Any) -> xr.Dataset:
            received.update(kwargs)
            return xr.Dataset()

        register_engine_loader(engine_name, dummy_loader)
        try:
            build_datacube(items=[], engine=engine_name, nodata=expected_nodata)
        finally:
            unregister_engine_loader(engine_name)

        self.assertEqual(received.get("nodata"), expected_nodata)

    def test_normalize_longitude_coordinate_wraps_range(self) -> None:
        ds = xr.Dataset(
            data_vars={
                "band": (
                    (DIM_TIME, DIM_LATITUDE, DIM_LONGITUDE),
                    np.zeros((1, 2, 5)),
                )
            },
            coords={
                DIM_TIME: ("time", pd.date_range("2020-01-01", periods=1)),
                DIM_LATITUDE: ("latitude", np.array([-10, 10])),
                DIM_LONGITUDE: ("longitude", np.array([0, 90, 180, 270, 350])),
            },
        )
        normalized = _normalize_longitude_coordinate(ds)

        np.testing.assert_allclose(
            normalized[DIM_LONGITUDE].values,
            np.array([-180, -90, -10, 0, 90]),
        )
        expected = ds.assign_coords({DIM_LONGITUDE: (((ds[DIM_LONGITUDE] + 180) % 360) - 180)}).sortby(DIM_LONGITUDE)
        self.assertTrue(normalized["band"].equals(expected["band"]))

    def test_normalize_longitude_coordinate_noop_for_standard_range(self) -> None:
        ds = xr.Dataset(
            coords={
                DIM_LONGITUDE: ("longitude", np.array([-120, 0, 45])),
            }
        )
        normalized = _normalize_longitude_coordinate(ds)
        np.testing.assert_array_equal(normalized[DIM_LONGITUDE], ds[DIM_LONGITUDE])

    def test_replace_item_hrefs_updates_assets(self) -> None:
        asset = FakeAsset(
            href="original",
            extra_fields={"alternate": {"download": {"href": "new-href"}}},
        )
        item = make_fake_item({"band": asset})
        _replace_item_hrefs([item], "alternate.download.href")
        self.assertEqual(item.assets["band"].href, "new-href")

    def test_replace_item_hrefs_handles_missing_path(self) -> None:
        asset = FakeAsset(href="original", extra_fields={"alternate": {}})
        item = make_fake_item({"band": asset})
        _replace_item_hrefs([item], "alternate.download.href")
        self.assertEqual(item.assets["band"].href, "original")

    @mock.patch("earthdaily.datacube._builder.geometry_to_geopandas", return_value="geo")
    @mock.patch("earthdaily.datacube._builder.bbox_to_geopandas")
    @mock.patch("earthdaily.datacube._builder.stac.load")
    def test_load_datacube_with_odc_assigns_metadata_and_coords(
        self, mock_stac_load, mock_bbox_to_geopandas, mock_geometry_to_geopandas
    ) -> None:
        mock_stac_load.return_value = self._make_dataset()
        items = self._make_items()

        result = _load_datacube_with_odc(
            items=items,
            assets={"band": "renamed_band"},
            bbox=[0, 0, 1, 1],
            intersects="shape",
            dtype="float64",
            nodata=-9999,
            properties=["quality"],
            replace_href_with="alternate.download.href",
            epsg=4326,
            resampling=int(Resampling.bilinear),
            chunks={"x": 2, "y": 2, "time": 1},
        )

        mock_geometry_to_geopandas.assert_called_once_with("shape")
        mock_bbox_to_geopandas.assert_not_called()

        _, kwargs = mock_stac_load.call_args
        self.assertEqual(kwargs["crs"], "EPSG:4326")
        self.assertEqual(kwargs["resampling"], Resampling.bilinear.name)
        self.assertEqual(kwargs["chunks"], {"x": 2, "y": 2, "time": 1})
        self.assertEqual(kwargs["nodata"], -9999)
        self.assertIn("geopolygon", kwargs)

        self.assertIn("renamed_band", result.data_vars)
        self.assertNotIn("band", result.data_vars)
        self.assertIn(DIM_Y, result.dims)
        self.assertIn(DIM_X, result.dims)
        np.testing.assert_allclose(result[DIM_X], np.array([-90.0, 0.0]))
        np.testing.assert_allclose(result.coords["quality"], np.array([0.3, 0.2]))

    @mock.patch("earthdaily.datacube._builder.bbox_to_geopandas", return_value="bbox-geo")
    @mock.patch("earthdaily.datacube._builder.stac.load")
    def test_load_datacube_with_odc_uses_bbox_when_intersects_missing(
        self, mock_stac_load, mock_bbox_to_geopandas
    ) -> None:
        mock_stac_load.return_value = self._make_dataset()
        item = self._make_items()[0]
        result = _load_datacube_with_odc(
            items=[item],
            bbox=[1, 2, 3, 4],
            properties=False,
        )
        mock_bbox_to_geopandas.assert_called_once_with([1, 2, 3, 4], crs=DEFAULT_BBOX_CRS)
        self.assertIn(DIM_X, result.coords)

    @mock.patch("earthdaily.datacube._builder.stac.load")
    def test_load_datacube_with_odc_skips_metadata_without_time_coord(self, mock_stac_load) -> None:
        data = xr.DataArray(
            np.zeros((1, 2)),
            dims=(DIM_LATITUDE, DIM_LONGITUDE),
            coords={DIM_LATITUDE: ("latitude", [0]), DIM_LONGITUDE: ("longitude", [0, 1])},
        )
        mock_stac_load.return_value = xr.Dataset({"renamed_band": data})
        item = self._make_items()[0]

        result = _load_datacube_with_odc(items=[item], properties=True, chunks={"x": "auto", "y": "auto"})
        self.assertNotIn("quality", result.coords)

    def test_load_datacube_with_odc_raises_without_items(self) -> None:
        with self.assertRaises(DatacubeCreationError):
            _load_datacube_with_odc(items=[])

    def test_load_datacube_with_odc_raises_for_missing_asset(self) -> None:
        items = self._make_items()
        with self.assertRaises(DatacubeCreationError):
            _load_datacube_with_odc(items=items, assets=["unknown"])

    @mock.patch("earthdaily.datacube._builder.stac.load", side_effect=ValueError("No such band/alias foo"))
    def test_load_datacube_with_odc_wraps_missing_band_error(self, mock_stac_load) -> None:
        items = self._make_items()
        with self.assertRaises(DatacubeCreationError) as ctx:
            _load_datacube_with_odc(items=items, assets=["band"])
        self.assertIn("Asset not found", str(ctx.exception))
        mock_stac_load.assert_called_once()

    @mock.patch("earthdaily.datacube._builder.stac.load", side_effect=ValueError("other error"))
    def test_load_datacube_with_odc_general_value_error(self, mock_stac_load) -> None:
        items = self._make_items()
        with self.assertRaises(DatacubeCreationError) as ctx:
            _load_datacube_with_odc(items=items)
        self.assertIn("Failed to create datacube", str(ctx.exception))

    @mock.patch("earthdaily.datacube._builder.stac.load", side_effect=RuntimeError("boom"))
    def test_load_datacube_with_odc_general_exception(self, mock_stac_load) -> None:
        items = self._make_items()
        with self.assertRaises(DatacubeCreationError) as ctx:
            _load_datacube_with_odc(items=items)
        self.assertIn("Unexpected error", str(ctx.exception))


class TestDeduplicateItems(unittest.TestCase):
    def _make_item(
        self,
        item_id: str,
        dt: datetime | None = None,
        collection_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> Item:
        item = Item(
            id=item_id,
            geometry=None,
            bbox=None,
            datetime=dt or datetime(2020, 1, 1),
            properties=properties or {},
        )
        item.collection_id = collection_id
        return item

    def test_deduplicate_by_date_keeps_last(self) -> None:
        items = [
            self._make_item("item-1", datetime(2020, 1, 1)),
            self._make_item("item-2", datetime(2020, 1, 1)),
            self._make_item("item-3", datetime(2020, 1, 2)),
        ]
        result = _deduplicate_items(items, deduplicate_by=["date"], deduplicate_keep="last")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "item-2")
        self.assertEqual(result[1].id, "item-3")

    def test_deduplicate_by_date_keeps_first(self) -> None:
        items = [
            self._make_item("item-1", datetime(2020, 1, 1)),
            self._make_item("item-2", datetime(2020, 1, 1)),
            self._make_item("item-3", datetime(2020, 1, 2)),
        ]
        result = _deduplicate_items(items, deduplicate_by=["date"], deduplicate_keep="first")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "item-1")
        self.assertEqual(result[1].id, "item-3")

    def test_deduplicate_by_collection_id(self) -> None:
        items = [
            self._make_item("item-1", collection_id="sentinel-2"),
            self._make_item("item-2", collection_id="sentinel-2"),
            self._make_item("item-3", collection_id="landsat"),
        ]
        result = _deduplicate_items(items, deduplicate_by=["collection_id"], deduplicate_keep="last")
        self.assertEqual(len(result), 2)

    def test_deduplicate_by_property(self) -> None:
        items = [
            self._make_item("item-1", properties={"proj:transform": "tile-A"}),
            self._make_item("item-2", properties={"proj:transform": "tile-A"}),
            self._make_item("item-3", properties={"proj:transform": "tile-B"}),
        ]
        result = _deduplicate_items(items, deduplicate_by=["proj:transform"], deduplicate_keep="last")
        self.assertEqual(len(result), 2)

    def test_deduplicate_by_multiple_keys(self) -> None:
        items = [
            self._make_item("item-1", datetime(2020, 1, 1), properties={"proj:transform": "tile-A"}),
            self._make_item("item-2", datetime(2020, 1, 1), properties={"proj:transform": "tile-A"}),
            self._make_item("item-3", datetime(2020, 1, 1), properties={"proj:transform": "tile-B"}),
            self._make_item("item-4", datetime(2020, 1, 2), properties={"proj:transform": "tile-A"}),
        ]
        result = _deduplicate_items(items, deduplicate_by=["date", "proj:transform"], deduplicate_keep="last")
        self.assertEqual(len(result), 3)

    def test_deduplicate_no_duplicates(self) -> None:
        items = [
            self._make_item("item-1", datetime(2020, 1, 1)),
            self._make_item("item-2", datetime(2020, 1, 2)),
            self._make_item("item-3", datetime(2020, 1, 3)),
        ]
        result = _deduplicate_items(items, deduplicate_by=["date"], deduplicate_keep="last")
        self.assertEqual(len(result), 3)

    def test_deduplicate_empty_list(self) -> None:
        result = _deduplicate_items([], deduplicate_by=["date"], deduplicate_keep="last")
        self.assertEqual(len(result), 0)

    def test_deduplicate_invalid_keep_raises(self) -> None:
        items = [self._make_item("item-1")]
        with self.assertRaises(ValueError) as ctx:
            _deduplicate_items(items, deduplicate_by=["date"], deduplicate_keep="invalid")
        self.assertIn("must be 'first' or 'last'", str(ctx.exception))

    def test_deduplicate_invalid_keep_raises_with_empty_list(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _deduplicate_items([], deduplicate_by=["date"], deduplicate_keep="invalid")
        self.assertIn("must be 'first' or 'last'", str(ctx.exception))

    def test_deduplicate_item_without_datetime_uses_id(self) -> None:
        item1 = Item(
            id="item-1",
            geometry=None,
            bbox=None,
            datetime=None,
            properties={"start_datetime": "2020-01-01T00:00:00Z", "end_datetime": "2020-01-01T23:59:59Z"},
        )
        item2 = Item(
            id="item-1",
            geometry=None,
            bbox=None,
            datetime=None,
            properties={"start_datetime": "2020-01-01T00:00:00Z", "end_datetime": "2020-01-01T23:59:59Z"},
        )
        items = [item1, item2]
        result = _deduplicate_items(items, deduplicate_by=["date"], deduplicate_keep="last")
        self.assertEqual(len(result), 1)

    def test_deduplicate_with_pipe_in_property_no_false_collision(self) -> None:
        items = [
            self._make_item("item-1", properties={"field1": "a|b", "field2": "c"}),
            self._make_item("item-2", properties={"field1": "a", "field2": "b|c"}),
        ]
        result = _deduplicate_items(items, deduplicate_by=["field1", "field2"], deduplicate_keep="last")
        self.assertEqual(len(result), 2)
        ids = {item.id for item in result}
        self.assertEqual(ids, {"item-1", "item-2"})


if __name__ == "__main__":
    unittest.main()
