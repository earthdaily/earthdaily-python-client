import unittest
from unittest.mock import MagicMock, patch

from earthdaily.platform.helpers.collections.sentinel1 import Sentinel1CollectionHelper


class TestSentinel1CollectionHelper(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.helper = Sentinel1CollectionHelper(self.mock_client)

    def test_class_constants_and_factory_methods(self):
        self.assertEqual(Sentinel1CollectionHelper.COLLECTION_ID, "sentinel-1-rtc")
        s1 = Sentinel1CollectionHelper.create_dual_pol(self.mock_client)
        self.assertEqual(s1.assets, ["vv", "vh"])
        s1_vv = Sentinel1CollectionHelper.create_single_pol_vv(self.mock_client)
        self.assertEqual(s1_vv.assets, ["vv"])
        s1_vh = Sentinel1CollectionHelper.create_single_pol_vh(self.mock_client)
        self.assertEqual(s1_vh.assets, ["vh"])
        s1_hh = Sentinel1CollectionHelper.create_single_pol_hh(self.mock_client)
        self.assertEqual(s1_hh.assets, ["hh"])

    def test_get_items_adds_instrument_mode_filter(self):
        core = self.helper._core
        with patch.object(core, "get_items", return_value=["i1"]) as gi:
            res = self.helper.get_items(instrument_mode="IW", max_items=3)
        self.assertEqual(res, ["i1"])
        called_kwargs = gi.call_args.kwargs
        self.assertEqual(called_kwargs["max_items"], 3)
        self.assertIn("query", called_kwargs)
        self.assertEqual(called_kwargs["query"], {"sar:instrument_mode": {"eq": "IW"}})

    def test_get_items_adds_platform_filter(self):
        core = self.helper._core
        with patch.object(core, "get_items", return_value=["i1"]) as gi:
            res = self.helper.get_items(platform="sentinel-1a", max_items=5)
        self.assertEqual(res, ["i1"])
        called_kwargs = gi.call_args.kwargs
        self.assertEqual(called_kwargs["query"], {"platform": {"eq": "sentinel-1a"}})

    def test_get_items_with_multiple_filters(self):
        core = self.helper._core
        with patch.object(core, "get_items", return_value=["i1"]) as gi:
            res = self.helper.get_items(instrument_mode="IW", platform="sentinel-1b")
        self.assertEqual(res, ["i1"])
        called_kwargs = gi.call_args.kwargs
        self.assertEqual(called_kwargs["query"]["sar:instrument_mode"], {"eq": "IW"})
        self.assertEqual(called_kwargs["query"]["platform"], {"eq": "sentinel-1b"})

    def test_create_datacube_with_filters(self):
        mock_datacube = MagicMock()
        with patch.object(self.helper._core, "create_datacube", return_value=mock_datacube) as cd:
            res = self.helper.create_datacube(assets=["vh"], instrument_mode="IW", max_items=4)
        self.assertEqual(res, mock_datacube)
        called_kwargs = cd.call_args.kwargs
        self.assertEqual(called_kwargs["assets"], ["vh"])
        self.assertEqual(called_kwargs["max_items"], 4)
        self.assertIn("search_kwargs", called_kwargs)
        self.assertEqual(called_kwargs["search_kwargs"]["query"], {"sar:instrument_mode": {"eq": "IW"}})

    def test_create_datacube_without_filters(self):
        mock_datacube = MagicMock()
        with patch.object(self.helper._core, "create_datacube", return_value=mock_datacube) as cd:
            res = self.helper.create_datacube(assets=["vv"])
        self.assertEqual(res, mock_datacube)
        called_kwargs = cd.call_args.kwargs
        self.assertEqual(called_kwargs["search_kwargs"], {})

    def test_polarization_and_orbit_helpers(self):
        item1 = MagicMock()
        item1.assets = {"vv": {}, "vh": {}}
        item1.properties = {
            "sat:orbit_state": "ascending",
            "sat:absolute_orbit": 1,
            "sat:relative_orbit": 2,
            "platform": "S1",
            "instruments": ["C-SAR"],
        }
        item1.id = "item1"

        item2 = MagicMock()
        item2.assets = {"vv": {}}
        item2.properties = {"sat:orbit_state": "descending"}
        item2.id = "item2"

        item3 = MagicMock()
        item3.assets = {"hh": {}}
        item3.properties = {"sat:orbit_state": "ascending"}
        item3.id = "item3"

        pol_info = self.helper.get_polarization_info([item1, item2, item3])
        self.assertEqual(pol_info["item1"], ["VV", "VH"])
        self.assertEqual(pol_info["item2"], ["VV"])
        self.assertEqual(pol_info["item3"], ["HH"])

        orbit_info = self.helper.get_orbit_info([item1])["item1"]
        self.assertEqual(orbit_info["orbit_direction"], "ascending")

        filtered = self.helper.filter_by_polarization([item1, item2, item3], ["VV", "VH"])
        self.assertEqual([i.id for i in filtered], ["item1"])

        filtered_hh = self.helper.filter_by_polarization([item1, item2, item3], ["HH"])
        self.assertEqual([i.id for i in filtered_hh], ["item3"])

    def test_asset_description(self):
        self.assertIn("VV polarization", self.helper.get_asset_description("vv"))
        self.assertIn("VH polarization", self.helper.get_asset_description("vh"))
        self.assertIn("HH polarization", self.helper.get_asset_description("hh"))
        self.assertIn("preview", self.helper.get_asset_description("rendered_preview").lower())
        self.assertIn("Asset:", self.helper.get_asset_description("unknown"))

    def test_get_available_assets_passthrough(self):
        with patch.object(self.helper._core, "get_available_assets", return_value=["vv", "vh"]) as gaa:
            assets = self.helper.get_available_assets()
        self.assertEqual(assets, ["vv", "vh"])
        gaa.assert_called_once_with()

    def test_download_assets_with_alternate_href(self):
        """Test downloading data assets that use alternate.download.href"""
        items = [{"id": "i"}]
        with patch.object(self.helper._core, "download_assets", return_value={"vv": "/tmp/vv.tif"}) as da:
            res = self.helper.download_assets(items=items, asset_keys=["vv"], output_dir="/tmp")
        self.assertEqual(res, {"vv": "/tmp/vv.tif"})
        da.assert_called_once()
        # Should NOT pass href_type for alternate href assets (uses default)
        call_kwargs = da.call_args.kwargs
        self.assertNotIn("href_type", call_kwargs)

    def test_download_assets_with_direct_href(self):
        """Test downloading preview assets that use direct href"""
        items = [{"id": "i"}]
        with patch.object(
            self.helper._core, "download_assets", return_value={"rendered_preview": "/tmp/preview.png"}
        ) as da:
            res = self.helper.download_assets(items=items, asset_keys=["rendered_preview"], output_dir="/tmp")
        self.assertEqual(res, {"rendered_preview": "/tmp/preview.png"})
        da.assert_called_once()
        # Should pass href_type="href" for direct href assets
        call_kwargs = da.call_args.kwargs
        self.assertEqual(call_kwargs.get("href_type"), "href")

    def test_download_assets_mixed_href_types(self):
        """Test downloading both data and preview assets together"""
        items = [{"id": "i"}]
        with patch.object(self.helper._core, "download_assets") as da:
            da.side_effect = [
                {"vv": "/tmp/vv.tif"},  # First call for alternate href assets
                {"rendered_preview": "/tmp/preview.png"},  # Second call for direct href assets
            ]
            res = self.helper.download_assets(items=items, asset_keys=["vv", "rendered_preview"], output_dir="/tmp")

        self.assertEqual(res, {"vv": "/tmp/vv.tif", "rendered_preview": "/tmp/preview.png"})
        self.assertEqual(da.call_count, 2)

        # First call should be for vv (no href_type)
        first_call_kwargs = da.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["asset_keys"], ["vv"])
        self.assertNotIn("href_type", first_call_kwargs)

        # Second call should be for rendered_preview (href_type="href")
        second_call_kwargs = da.call_args_list[1].kwargs
        self.assertEqual(second_call_kwargs["asset_keys"], ["rendered_preview"])
        self.assertEqual(second_call_kwargs["href_type"], "href")

    def test_repr(self):
        repr_str = repr(self.helper)
        self.assertIn("Sentinel1CollectionHelper", repr_str)
        self.assertIn("sentinel-1-rtc", repr_str)


if __name__ == "__main__":
    unittest.main()
