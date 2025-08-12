import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pystac

from earthdaily._eds_config import AssetAccessMode
from earthdaily.exceptions import EDSAPIError
from earthdaily.platform._stac_item import StacItemService


class TestStacItemServiceDownloadAssets(unittest.TestCase):
    def setUp(self):
        self.mock_api_requester = Mock()
        self.mock_api_requester.base_url = "https://api.example.com"
        self.mock_config = Mock()
        self.mock_config.asset_access_mode = AssetAccessMode.PRESIGNED_URLS
        self.mock_api_requester.config = self.mock_config
        self.stac_item_service = StacItemService(self.mock_api_requester)

        # Create a sample STAC item for testing
        self.sample_stac_item_dict = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "test_item_id",
            "collection": "test_collection",
            "properties": {"datetime": "2023-01-01T00:00:00Z"},
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "assets": {"visual": {"href": "https://example.com/test_asset.tif", "type": "image/tiff"}},
        }

        # Mock return path for downloaded assets
        self.mock_downloaded_path = {"visual": Path("/mock/path/test_asset.tif")}

    @patch("earthdaily.platform._stac_item.ItemDownloader")
    def test_download_assets_init_downloader(self, mock_downloader_cls):
        """Test that ItemDownloader is initialized with correct parameters"""
        mock_downloader_instance = Mock()
        mock_downloader_cls.return_value = mock_downloader_instance
        mock_downloader_instance.download_assets.return_value = self.mock_downloaded_path

        # Call with custom parameters (only use max_workers which is supported)
        self.stac_item_service.download_assets(item=self.sample_stac_item_dict, max_workers=8)

        # Verify the downloader was initialized with correct parameters
        mock_downloader_cls.assert_called_once_with(max_workers=8, api_requester=self.mock_api_requester)

    @patch("earthdaily.platform._stac_item.ItemDownloader")
    def test_download_assets_init_downloader_with_proxy_urls(self, mock_downloader_cls):
        """Test that ItemDownloader automatically detects proxy URLs from config"""
        # Set up config for proxy URLs
        self.mock_config.asset_access_mode = AssetAccessMode.PROXY_URLS

        mock_downloader_instance = Mock()
        mock_downloader_cls.return_value = mock_downloader_instance
        mock_downloader_instance.download_assets.return_value = self.mock_downloaded_path

        self.stac_item_service.download_assets(item=self.sample_stac_item_dict, max_workers=4)

        # Verify the downloader was initialized with correct parameters
        mock_downloader_cls.assert_called_once_with(max_workers=4, api_requester=self.mock_api_requester)

    @patch("earthdaily.platform._stac_item.ItemDownloader")
    def test_download_assets_with_proxy_urls_and_custom_href_type(self, mock_downloader_cls):
        """Test downloading assets with proxy URLs (auto-detected) and custom href_type"""
        # Set up config for proxy URLs
        self.mock_config.asset_access_mode = AssetAccessMode.PROXY_URLS

        mock_downloader_instance = Mock()
        mock_downloader_cls.return_value = mock_downloader_instance
        mock_downloaded_path = {"visual": Path("/mock/path/alternate_asset.tif")}
        mock_downloader_instance.download_assets.return_value = mock_downloaded_path

        result = self.stac_item_service.download_assets(
            item=self.sample_stac_item_dict,
            asset_keys=["visual"],
            output_dir="/test/output",
            href_type="alternate.download.href",
        )

        # Verify the downloader was initialized
        mock_downloader_cls.assert_called_once_with(max_workers=3, api_requester=self.mock_api_requester)

        # Verify the download_assets call included custom href_type
        mock_downloader_instance.download_assets.assert_called_once_with(
            item=self.sample_stac_item_dict,
            asset_keys=["visual"],
            output_dir="/test/output",
            quiet=False,
            continue_on_error=True,
            href_type="alternate.download.href",
        )

        self.assertEqual(result, mock_downloaded_path)

    @patch("earthdaily.platform._stac_item.ItemDownloader")
    def test_download_assets_proxy_urls_with_string_item(self, mock_downloader_cls):
        """Test downloading assets with proxy URLs (auto-detected) when item is provided as string"""
        # Set up config for proxy URLs
        self.mock_config.asset_access_mode = AssetAccessMode.PROXY_URLS

        mock_downloader_instance = Mock()
        mock_downloader_cls.return_value = mock_downloader_instance
        mock_downloader_instance.download_assets.return_value = self.mock_downloaded_path

        with patch.object(self.stac_item_service, "get_item", return_value=self.sample_stac_item_dict) as mock_get_item:
            result = self.stac_item_service.download_assets(item="test_collection/test_item_id")

        # Verify get_item was called
        mock_get_item.assert_called_once_with(
            collection_id="test_collection", item_id="test_item_id", return_format="dict"
        )

        # Verify downloader was initialized
        mock_downloader_cls.assert_called_once_with(max_workers=3, api_requester=self.mock_api_requester)

        self.assertEqual(result, self.mock_downloaded_path)

    @patch("earthdaily.platform._stac_item.ItemDownloader")
    def test_download_assets_with_presigned_urls_config(self, mock_downloader_cls):
        """Test that ItemDownloader works with presigned URLs config"""
        # Config is already set to PRESIGNED_URLS in setUp
        mock_downloader_instance = Mock()
        mock_downloader_cls.return_value = mock_downloader_instance
        mock_downloader_instance.download_assets.return_value = self.mock_downloaded_path

        self.stac_item_service.download_assets(item=self.sample_stac_item_dict)

        # Verify the downloader was initialized correctly
        mock_downloader_cls.assert_called_once_with(max_workers=3, api_requester=self.mock_api_requester)

    @patch("earthdaily.platform._stac_item_downloader.ItemDownloader.download_assets")
    def test_download_assets_with_dict(self, mock_download):
        """Test downloading assets when providing a dictionary item"""
        mock_download.return_value = self.mock_downloaded_path

        result = self.stac_item_service.download_assets(
            item=self.sample_stac_item_dict, asset_keys=["visual"], output_dir="/test/output"
        )

        # Verify the downloader was called with correct arguments
        mock_download.assert_called_once_with(
            item=self.sample_stac_item_dict,
            asset_keys=["visual"],
            output_dir="/test/output",
            quiet=False,
            continue_on_error=True,
            href_type="alternate.download.href",
        )

        # Verify the result is passed through from the downloader
        self.assertEqual(result, self.mock_downloaded_path)

    @patch("earthdaily.platform._stac_item_downloader.ItemDownloader.download_assets")
    def test_download_assets_with_pystac_item(self, mock_download):
        """Test downloading assets when providing a pystac.Item object"""
        mock_download.return_value = self.mock_downloaded_path

        # Create a pystac Item
        pystac_item = pystac.Item.from_dict(self.sample_stac_item_dict)

        result = self.stac_item_service.download_assets(
            item=pystac_item, asset_keys=["visual"], output_dir="/test/output"
        )

        # Verify the downloader was called with correct arguments
        mock_download.assert_called_once()
        # The first argument (item) should be the pystac_item
        args, kwargs = mock_download.call_args
        self.assertEqual(kwargs["item"], pystac_item)

        # Verify the result is passed through from the downloader
        self.assertEqual(result, self.mock_downloaded_path)

    @patch("earthdaily.platform._stac_item.StacItemService.get_item")
    @patch("earthdaily.platform._stac_item_downloader.ItemDownloader.download_assets")
    def test_download_assets_with_item_string(self, mock_download, mock_get_item):
        """Test downloading assets when providing a string in format 'collection_id/item_id'"""
        mock_get_item.return_value = self.sample_stac_item_dict
        mock_download.return_value = self.mock_downloaded_path

        result = self.stac_item_service.download_assets(
            item="test_collection/test_item_id", asset_keys=["visual"], output_dir="/test/output"
        )

        # Verify get_item was called correctly
        mock_get_item.assert_called_once_with(
            collection_id="test_collection", item_id="test_item_id", return_format="dict"
        )

        # Verify the downloader was called with correct arguments
        mock_download.assert_called_once_with(
            item=self.sample_stac_item_dict,
            asset_keys=["visual"],
            output_dir="/test/output",
            quiet=False,
            continue_on_error=True,
            href_type="alternate.download.href",
        )

        # Verify the result is passed through from the downloader
        self.assertEqual(result, self.mock_downloaded_path)

    def test_download_assets_invalid_string_format(self):
        """Test that an invalid item string format raises a ValueError"""
        with self.assertRaises(ValueError) as context:
            self.stac_item_service.download_assets(item="invalid_format")

        self.assertIn("must be in the format 'collection_id/item_id'", str(context.exception))

    def test_download_assets_invalid_item_type(self):
        """Test that an invalid item type raises a ValueError"""
        with self.assertRaises(ValueError) as context:
            self.stac_item_service.download_assets(item=123)  # Integer is not a valid type

        self.assertIn("Item must be a PySTAC Item or a dictionary", str(context.exception))

    @patch("earthdaily.platform._stac_item.StacItemService.get_item")
    def test_download_assets_api_error(self, mock_get_item):
        """Test handling API error when fetching item by string reference"""
        # Setup get_item to raise an EDSAPIError
        mock_get_item.side_effect = EDSAPIError("API Error: Item not found")

        with self.assertRaises(EDSAPIError) as context:
            self.stac_item_service.download_assets(item="test_collection/test_item_id")

        self.assertIn("API Error: Item not found", str(context.exception))

    @patch("earthdaily.platform._stac_item.StacItemService.get_item")
    @patch("earthdaily.platform._stac_item_downloader.ItemDownloader.download_assets")
    def test_download_assets_custom_parameters(self, mock_download, mock_get_item):
        """Test downloading assets with custom parameters"""
        mock_get_item.return_value = self.sample_stac_item_dict
        mock_download.return_value = self.mock_downloaded_path

        self.stac_item_service.download_assets(
            item="test_collection/test_item_id",
            asset_keys=["visual"],
            output_dir="/custom/output",
            max_workers=5,
            quiet=True,
            continue_on_error=False,
            href_type="alternate",
        )

        # Verify ItemDownloader was initialized with max_workers=5
        mock_download.assert_called_once()
        # The downloader call should include all the custom parameters
        args, kwargs = mock_download.call_args
        self.assertEqual(kwargs["output_dir"], "/custom/output")
        self.assertEqual(kwargs["quiet"], True)
        self.assertEqual(kwargs["continue_on_error"], False)
        self.assertEqual(kwargs["href_type"], "alternate")

    @patch("earthdaily.platform._stac_item_downloader.ItemDownloader.download_assets")
    def test_download_assets_without_asset_keys(self, mock_download):
        """Test downloading assets without specifying asset_keys (all assets)"""
        mock_download.return_value = self.mock_downloaded_path

        result = self.stac_item_service.download_assets(item=self.sample_stac_item_dict, output_dir="/test/output")

        # Verify the downloader was called with None for asset_keys
        mock_download.assert_called_once()
        args, kwargs = mock_download.call_args
        self.assertIsNone(kwargs["asset_keys"])

        # Verify the result is passed through from the downloader
        self.assertEqual(result, self.mock_downloaded_path)


if __name__ == "__main__":
    unittest.main()
