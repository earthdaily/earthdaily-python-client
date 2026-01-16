import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from earthdaily.platform.helpers.collections.base import CollectionHelper, SpatioTemporalGeometry


class TestSTGeometry(unittest.TestCase):
    def test_to_dict(self):
        with patch("earthdaily.platform.helpers.collections.base.shapely.to_geojson", return_value={"type": "Point"}):
            geom_obj = SimpleNamespace(__geo_interface__={"type": "Point", "coordinates": [0, 0]})
            now = datetime.now()
            earlier = now - timedelta(days=1)
            st = SpatioTemporalGeometry(crs="EPSG:4326", geometry=geom_obj, time_range=(earlier, now))
            d = st.to_dict()
            self.assertEqual(d["crs"], "EPSG:4326")
            self.assertEqual(d["geometry"], {"type": "Point"})
            self.assertEqual(d["time_range"], [earlier.isoformat(), now.isoformat()])


class TestCollectionHelper(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.helper = CollectionHelper(
            client=self.mock_client, collection_id="collection-x", asset_mapping={"a": ["A"]}
        )

    def test_get_items_with_geometry_and_datetime(self):
        search_mock = MagicMock()
        items_list = [Mock(id="i1"), Mock(id="i2")]
        search_mock.items.return_value = items_list
        self.mock_client.platform.pystac_client.search.return_value = search_mock

        geom_obj = SimpleNamespace(__geo_interface__={"type": "Point", "coordinates": [0, 0]})
        now = datetime.now()
        earlier = now - timedelta(days=7)
        st = SpatioTemporalGeometry(crs="EPSG:4326", geometry=geom_obj, time_range=(earlier, now))

        result = self.helper.get_items(geometries=[st], max_items=5, additional="x")

        self.assertEqual(result, items_list)
        called_kwargs = self.mock_client.platform.pystac_client.search.call_args.kwargs
        self.assertEqual(called_kwargs["collections"], ["collection-x"])
        self.assertEqual(called_kwargs["max_items"], 5)
        self.assertEqual(called_kwargs["intersects"], geom_obj.__geo_interface__)
        self.assertIn("datetime", called_kwargs)
        self.assertIn("additional", called_kwargs)

    def test_create_datacube_with_items(self):
        mock_datacube = MagicMock()
        self.mock_client.datacube.create.return_value = mock_datacube

        items = [MagicMock(id="a")]
        res = self.helper.create_datacube(items=items, assets=None)

        self.assertEqual(res, mock_datacube)
        self.mock_client.datacube.create.assert_called_once()
        called_kwargs = self.mock_client.datacube.create.call_args.kwargs
        self.assertEqual(called_kwargs["items"], items)
        self.assertEqual(called_kwargs["assets"], ["a"])

    def test_create_datacube_search_then_build(self):
        mock_datacube = MagicMock()
        self.mock_client.datacube.create.return_value = mock_datacube

        with patch.object(self.helper, "get_items", return_value=[{"id": "b"}]) as gi:
            res = self.helper.create_datacube(assets=["a"], max_items=2)

        gi.assert_called_once()
        self.assertEqual(res, mock_datacube)
        called_kwargs = self.mock_client.datacube.create.call_args.kwargs
        self.assertEqual(called_kwargs["assets"], ["a"])

    def test_download_assets_aggregates_and_continues_on_error(self):
        downloader = self.mock_client.platform.stac_item.download_assets
        downloader.side_effect = [
            {"visual": "/tmp/visual.tif"},
            Exception("x"),
        ]
        items = [
            {"id": "i1", "assets": {"visual": {"href": "u"}}},
            {"id": "i2", "assets": {"visual": {"href": "v"}}},
        ]
        res = self.helper.download_assets(items=items, asset_keys=["visual"], output_dir="/tmp")
        self.assertEqual(res, {"visual": "/tmp/visual.tif"})

    def test_get_available_assets(self):
        assets = self.helper.get_available_assets()
        self.assertEqual(assets, ["a"])

    def test_get_asset_description(self):
        desc = self.helper.get_asset_description("a")
        self.assertIn("Asset 'a'", desc)
        self.assertIn("A", desc)


if __name__ == "__main__":
    unittest.main()
