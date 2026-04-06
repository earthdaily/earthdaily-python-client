from __future__ import annotations

import unittest
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

from pystac import Asset, Item

from earthdaily.datacube._enrichment import RasterMetadataEnricher


def _make_mock_dataset(
    *,
    epsg: int | None = 32631,
    width: int = 256,
    height: int = 256,
    res: float = 10.0,
    nodata: float | None = 0.0,
    count: int = 1,
    dtypes: tuple[str, ...] | None = None,
    crs_is_none: bool = False,
    gcps: tuple | None = None,
) -> MagicMock:
    ds = MagicMock()
    if crs_is_none:
        ds.crs = None
        if gcps:
            ds.gcps = gcps
        else:
            ds.gcps = ([], None)
    else:
        crs_mock = MagicMock()
        crs_mock.to_epsg.return_value = epsg
        crs_mock.to_wkt.return_value = "PROJCS[...]"
        ds.crs = crs_mock

    transform = MagicMock()
    transform.a = res
    transform.b = 0.0
    transform.c = 500000.0
    transform.d = 0.0
    transform.e = -res
    transform.f = 6000000.0
    ds.transform = transform

    ds.width = width
    ds.height = height
    ds.count = count
    ds.dtypes = dtypes or tuple(["uint16"] * count)
    ds.nodata = nodata
    ds.units = None
    ds.scales = None
    ds.offsets = None

    return ds


def _make_item(
    item_id: str = "test-item",
    assets: dict[str, dict[str, Any]] | None = None,
    stac_extensions: list[str] | None = None,
    properties: dict[str, Any] | None = None,
) -> Item:
    default_assets = {
        "B02": {
            "href": "https://example.com/B02.tif",
            "type": "image/tiff; application=geotiff",
        }
    }
    item = Item(
        id=item_id,
        geometry={"type": "Point", "coordinates": [0, 0]},
        bbox=[0, 0, 1, 1],
        datetime=datetime(2024, 1, 1),
        properties=properties or {},
        stac_extensions=stac_extensions or [],
    )
    for key, asset_dict in (assets or default_assets).items():
        extra = {k: v for k, v in asset_dict.items() if k not in ("href", "type", "roles", "title")}
        item.add_asset(
            key,
            Asset(
                href=asset_dict.get("href", "https://example.com/test.tif"),
                media_type=asset_dict.get("type"),
                extra_fields=extra,
            ),
        )
    return item


class TestEnrichBasic(unittest.TestCase):
    """Items that lack proj/raster metadata get populated."""

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_enriches_missing_metadata(self, mock_rasterio):
        mock_ds = _make_mock_dataset()
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        item = _make_item()
        result = RasterMetadataEnricher.enrich_item(item)

        self.assertIsInstance(result, Item)

        asset = result.assets["B02"]
        self.assertEqual(asset.extra_fields["proj:code"], "EPSG:32631")
        self.assertEqual(asset.extra_fields["proj:shape"], [256, 256])
        self.assertAlmostEqual(asset.extra_fields["gsd"], 10.0)
        self.assertEqual(len(asset.extra_fields["proj:transform"]), 6)
        self.assertIn("raster:bands", asset.extra_fields)
        self.assertEqual(len(asset.extra_fields["raster:bands"]), 1)
        self.assertEqual(asset.extra_fields["raster:bands"][0]["data_type"], "uint16")


class TestEnrichSkipsExisting(unittest.TestCase):
    """Items with existing metadata are skipped when force=False."""

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_skips_when_metadata_present(self, mock_rasterio):
        item = _make_item(
            assets={
                "B02": {
                    "href": "https://example.com/B02.tif",
                    "type": "image/tiff; application=geotiff",
                    "proj:code": "EPSG:32631",
                    "proj:transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0],
                    "proj:shape": [256, 256],
                    "gsd": 10.0,
                    "raster:bands": [{"data_type": "uint16"}],
                },
            },
        )

        result = RasterMetadataEnricher.enrich_item(item, force=False)
        mock_rasterio.open.assert_not_called()

        self.assertEqual(result.assets["B02"].extra_fields["proj:code"], "EPSG:32631")


class TestEnrichForce(unittest.TestCase):
    """Existing metadata is overwritten when force=True."""

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_force_overwrites_existing(self, mock_rasterio):
        mock_ds = _make_mock_dataset(epsg=4326, res=0.001)
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        item = _make_item(
            assets={
                "B02": {
                    "href": "https://example.com/B02.tif",
                    "type": "image/tiff; application=geotiff",
                    "proj:code": "EPSG:32631",
                    "gsd": 10.0,
                    "raster:bands": [{"data_type": "float32"}],
                },
            },
        )

        result = RasterMetadataEnricher.enrich_item(item, force=True)
        self.assertEqual(result.assets["B02"].extra_fields["proj:code"], "EPSG:4326")
        self.assertAlmostEqual(result.assets["B02"].extra_fields["gsd"], 0.001)
        self.assertEqual(result.assets["B02"].extra_fields["raster:bands"][0]["data_type"], "uint16")


class TestEnrichSkipsNonTiff(unittest.TestCase):
    """Non-tiff assets are skipped."""

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_non_tiff_assets_ignored(self, mock_rasterio):
        item = _make_item(
            assets={
                "thumbnail": {
                    "href": "https://example.com/thumb.png",
                    "type": "image/png",
                },
                "metadata": {
                    "href": "https://example.com/metadata.json",
                    "type": "application/json",
                },
            }
        )

        RasterMetadataEnricher.enrich_item(item)
        mock_rasterio.open.assert_not_called()


class TestEnrichHandlesFailures(unittest.TestCase):
    """rasterio failures log a warning and continue."""

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_rasterio_failure_logs_and_continues(self, mock_rasterio):
        mock_rasterio.open.side_effect = Exception("Connection refused")

        item = _make_item()
        result = RasterMetadataEnricher.enrich_item(item)

        self.assertNotIn("proj:code", result.assets["B02"].extra_fields)

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_partial_failure_enriches_others(self, mock_rasterio):
        mock_ds = _make_mock_dataset()

        call_count = 0

        def side_effect(href):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Timeout")
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=mock_ds)
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        mock_rasterio.open.side_effect = side_effect

        item1 = _make_item(item_id="fail-item")
        item2 = _make_item(item_id="ok-item")
        result1 = RasterMetadataEnricher.enrich_item(item1)
        result2 = RasterMetadataEnricher.enrich_item(item2)

        self.assertNotIn("proj:code", result1.assets["B02"].extra_fields)
        self.assertEqual(result2.assets["B02"].extra_fields["proj:code"], "EPSG:32631")


class TestItemPropertyPromotion(unittest.TestCase):
    """Item-level properties are promoted when all assets agree."""

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_properties_promoted_when_uniform(self, mock_rasterio):
        mock_ds = _make_mock_dataset(epsg=32631, res=10.0)
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        item = _make_item(
            assets={
                "B02": {
                    "href": "https://example.com/B02.tif",
                    "type": "image/tiff; application=geotiff",
                },
                "B03": {
                    "href": "https://example.com/B03.tif",
                    "type": "image/tiff; application=geotiff",
                },
            }
        )
        result = RasterMetadataEnricher.enrich_item(item)

        self.assertAlmostEqual(result.properties["gsd"], 10.0)
        self.assertEqual(result.properties["proj:code"], "EPSG:32631")

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_properties_not_promoted_when_mixed(self, mock_rasterio):
        call_count = 0

        def side_effect(href):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                ds = _make_mock_dataset(epsg=32631, res=10.0)
            else:
                ds = _make_mock_dataset(epsg=4326, res=0.001)
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ds)
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        mock_rasterio.open.side_effect = side_effect

        item = _make_item(
            assets={
                "B02": {
                    "href": "https://example.com/B02.tif",
                    "type": "image/tiff; application=geotiff",
                },
                "B03": {
                    "href": "https://example.com/B03.tif",
                    "type": "image/tiff; application=geotiff",
                },
            }
        )
        result = RasterMetadataEnricher.enrich_item(item)

        self.assertNotIn("proj:code", result.properties)


class TestWkt2Fallback(unittest.TestCase):
    """CRS without EPSG falls back to WKT2."""

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_wkt2_when_no_epsg(self, mock_rasterio):
        mock_ds = _make_mock_dataset(epsg=None)
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        item = _make_item()
        result = RasterMetadataEnricher.enrich_item(item)

        self.assertIn("proj:wkt2", result.assets["B02"].extra_fields)
        self.assertNotIn("proj:code", result.assets["B02"].extra_fields)


class TestNoCrs(unittest.TestCase):
    """Dataset with no CRS and no GCPs returns None metadata."""

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_no_crs_no_gcps(self, mock_rasterio):
        mock_ds = _make_mock_dataset(crs_is_none=True)
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        item = _make_item()
        result = RasterMetadataEnricher.enrich_item(item)

        self.assertNotIn("proj:code", result.assets["B02"].extra_fields)


class TestRasterBandDetails(unittest.TestCase):
    """Raster band metadata captures nodata, scale, offset, units."""

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_band_metadata_captured(self, mock_rasterio):
        mock_ds = _make_mock_dataset(nodata=-9999.0, count=2, dtypes=("float32", "float32"))
        mock_ds.units = ("reflectance", "reflectance")
        mock_ds.scales = (0.0001, 0.0001)
        mock_ds.offsets = (-0.1, -0.1)
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        item = _make_item()
        result = RasterMetadataEnricher.enrich_item(item)

        bands = result.assets["B02"].extra_fields["raster:bands"]
        self.assertEqual(len(bands), 2)
        self.assertEqual(bands[0]["data_type"], "float32")
        self.assertEqual(bands[0]["nodata"], -9999.0)
        self.assertEqual(bands[0]["unit"], "reflectance")
        self.assertEqual(bands[0]["scale"], 0.0001)
        self.assertEqual(bands[0]["offset"], -0.1)


class TestReturnsNewItem(unittest.TestCase):
    """Input item is not mutated."""

    @patch("earthdaily.datacube._enrichment.rasterio")
    def test_input_not_mutated(self, mock_rasterio):
        mock_ds = _make_mock_dataset()
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        item = _make_item()
        original_dict = item.to_dict()

        RasterMetadataEnricher.enrich_item(item)

        self.assertEqual(item.to_dict(), original_dict)


if __name__ == "__main__":
    unittest.main()
