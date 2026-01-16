import unittest
from unittest.mock import MagicMock, patch

from earthdaily.platform.helpers.collections.sentinel2 import Sentinel2CollectionHelper


class TestSentinel2CollectionHelper(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.helper = Sentinel2CollectionHelper(self.mock_client)

    def test_class_constants_and_factories(self):
        self.assertEqual(Sentinel2CollectionHelper.COLLECTION_ID, "sentinel-2-l2a")
        rgb = Sentinel2CollectionHelper.create_rgb(self.mock_client)
        self.assertEqual(rgb.assets, ["red", "green", "blue"])
        veg = Sentinel2CollectionHelper.create_vegetation(self.mock_client)
        self.assertTrue(set(["nir", "red"]).issubset(set(veg.assets)))
        ag = Sentinel2CollectionHelper.create_agriculture(self.mock_client)
        self.assertIn("swir16", ag.assets)
        ar = Sentinel2CollectionHelper.create_analysis_ready(self.mock_client)
        self.assertEqual(set(ar.assets), set(["blue", "green", "red", "nir", "swir16", "swir22"]))

    def test_get_items_adds_cloud_filter(self):
        with patch.object(self.helper._core, "get_items", return_value=["i"]) as gi:
            res = self.helper.get_items(cloud_cover_max=12.5, max_items=7)
        self.assertEqual(res, ["i"])
        called_kwargs = gi.call_args.kwargs
        self.assertEqual(called_kwargs["max_items"], 7)
        self.assertIn("query", called_kwargs)
        self.assertEqual(called_kwargs["query"], {"eo:cloud_cover": {"lt": 12.5}})

    def test_create_datacube_with_cloud_mask(self):
        mock_datacube = MagicMock()
        mock_datacube.bands = ["blue", "scl"]
        mock_masked_datacube = MagicMock()
        mock_datacube.apply_mask.return_value = mock_masked_datacube

        with patch.object(self.helper._core, "create_datacube", return_value=mock_datacube) as cd:
            res = self.helper.create_datacube(
                geometries=None,
                assets=["blue"],
                cloud_cover_max=5,
                apply_cloud_mask=True,
                max_items=9,
            )

        self.assertEqual(res, mock_masked_datacube)
        called_kwargs = cd.call_args.kwargs
        self.assertIn("blue", called_kwargs["assets"])
        self.assertIn("scl", called_kwargs["assets"])
        self.assertEqual(called_kwargs["max_items"], 9)
        self.assertEqual(called_kwargs["search_kwargs"]["query"], {"eo:cloud_cover": {"lt": 5}})

        mock_datacube.apply_mask.assert_called_once()
        mask_call_kwargs = mock_datacube.apply_mask.call_args.kwargs
        self.assertEqual(mask_call_kwargs["mask_band"], "scl")
        self.assertEqual(mask_call_kwargs["exclude_values"], Sentinel2CollectionHelper.DEFAULT_EXCLUDE_VALUES)

    def test_create_datacube_without_cloud_mask(self):
        mock_datacube = MagicMock()
        mock_datacube.bands = ["blue"]

        with patch.object(self.helper._core, "create_datacube", return_value=mock_datacube) as cd:
            res = self.helper.create_datacube(
                assets=["blue"],
                apply_cloud_mask=False,
            )

        self.assertEqual(res, mock_datacube)
        mock_datacube.apply_mask.assert_not_called()
        called_kwargs = cd.call_args.kwargs
        self.assertEqual(called_kwargs["assets"], ["blue"])

    def test_create_datacube_custom_exclude_values(self):
        mock_datacube = MagicMock()
        mock_datacube.bands = ["blue", "scl"]
        mock_masked_datacube = MagicMock()
        mock_datacube.apply_mask.return_value = mock_masked_datacube

        with patch.object(self.helper._core, "create_datacube", return_value=mock_datacube):
            self.helper.create_datacube(
                assets=["blue"],
                apply_cloud_mask=True,
                exclude_values=[1, 2, 3],
            )

        mask_call_kwargs = mock_datacube.apply_mask.call_args.kwargs
        self.assertEqual(mask_call_kwargs["exclude_values"], [1, 2, 3])

    def test_cloud_and_processing_helpers(self):
        item = MagicMock()
        item.properties = {
            "eo:cloud_cover": 7.5,
            "processing:level": "L2A",
            "s2:processing_baseline": "05.09",
            "s2:product_type": "S2MSI2A",
            "s2:datatake_id": "D",
            "s2:granule_id": "G",
            "s2:mgrs_tile": "11SPC",
            "platform": "S2A",
            "instruments": ["MSI"],
        }
        item.id = "it"
        cloud_info = self.helper.get_cloud_cover_info([item])
        self.assertEqual(cloud_info["it"], 7.5)
        proc = self.helper.get_processing_info([item])["it"]
        self.assertEqual(proc["processing_level"], "L2A")

    def test_filter_by_cloud_cover(self):
        item1 = MagicMock()
        item1.properties = {"eo:cloud_cover": 3}
        item1.id = "i1"
        item2 = MagicMock()
        item2.properties = {"eo:cloud_cover": 20}
        item2.id = "i2"
        res = self.helper.filter_by_cloud_cover([item1, item2], 10)
        self.assertEqual([i.id for i in res], ["i1"])

    def test_calculate_ndvi_requires_bands(self):
        mock_datacube = MagicMock()
        mock_datacube.bands = ["blue", "green"]
        with self.assertRaises(ValueError) as ctx:
            self.helper.calculate_ndvi(mock_datacube)
        self.assertIn("red", str(ctx.exception))
        self.assertIn("nir", str(ctx.exception))

    def test_calculate_ndwi_requires_bands(self):
        mock_datacube = MagicMock()
        mock_datacube.bands = ["red", "blue"]
        with self.assertRaises(ValueError) as ctx:
            self.helper.calculate_ndwi(mock_datacube)
        self.assertIn("green", str(ctx.exception))
        self.assertIn("nir", str(ctx.exception))

    def test_calculate_ndvi_success(self):
        mock_datacube = MagicMock()
        mock_datacube.bands = ["red", "nir"]
        mock_datacube.data = {"red": MagicMock(), "nir": MagicMock()}
        mock_result = MagicMock()
        mock_datacube.add_indices.return_value = mock_result

        result = self.helper.calculate_ndvi(mock_datacube)

        self.assertEqual(result, mock_result)
        mock_datacube.add_indices.assert_called_once()
        call_args = mock_datacube.add_indices.call_args
        self.assertEqual(call_args[0][0], ["NDVI"])

    def test_calculate_ndwi_success(self):
        mock_datacube = MagicMock()
        mock_datacube.bands = ["green", "nir"]
        mock_datacube.data = {"green": MagicMock(), "nir": MagicMock()}
        mock_result = MagicMock()
        mock_datacube.add_indices.return_value = mock_result

        result = self.helper.calculate_ndwi(mock_datacube)

        self.assertEqual(result, mock_result)
        mock_datacube.add_indices.assert_called_once()
        call_args = mock_datacube.add_indices.call_args
        self.assertEqual(call_args[0][0], ["NDWI"])

    def test_get_band_info(self):
        blue_info = Sentinel2CollectionHelper.get_band_info("blue")
        self.assertEqual(blue_info["band_id"], "B02")
        self.assertEqual(blue_info["resolution"], 10)

        unknown_info = Sentinel2CollectionHelper.get_band_info("unknown")
        self.assertIn("unknown", unknown_info["description"])

    def test_get_available_assets_and_download_passthrough(self):
        with patch.object(self.helper._core, "get_available_assets", return_value=["red"]) as gaa:
            assets = self.helper.get_available_assets()
        self.assertEqual(assets, ["red"])
        gaa.assert_called_once_with()

        items = [{"id": "it"}]
        with patch.object(self.helper._core, "download_assets", return_value={"red": "/tmp/red.tif"}) as da:
            res = self.helper.download_assets(items=items, asset_keys=["red"], output_dir="/tmp")
        self.assertEqual(res, {"red": "/tmp/red.tif"})
        da.assert_called_once()

    def test_repr(self):
        repr_str = repr(self.helper)
        self.assertIn("Sentinel2CollectionHelper", repr_str)
        self.assertIn("sentinel-2-l2a", repr_str)

    def test_scl_constants(self):
        self.assertEqual(Sentinel2CollectionHelper.SCL_CLOUD_SHADOW, 3)
        self.assertEqual(Sentinel2CollectionHelper.SCL_CLOUD_MEDIUM, 8)
        self.assertEqual(Sentinel2CollectionHelper.SCL_CLOUD_HIGH, 9)
        self.assertEqual(Sentinel2CollectionHelper.SCL_THIN_CIRRUS, 10)
        self.assertEqual(Sentinel2CollectionHelper.DEFAULT_EXCLUDE_VALUES, [3, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
