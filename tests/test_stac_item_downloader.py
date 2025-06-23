import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pystac

from earthdaily.platform._stac_item_downloader import CustomHeadersDownloader, ItemDownloader


class TestItemDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = ItemDownloader(max_workers=3, timeout=60, allow_redirects=True)

        self.sample_item_dict = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "test_item",
            "properties": {"datetime": "2023-01-01T00:00:00Z"},
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "assets": {
                "visual": {"href": "https://example.com/visual.tif", "type": "image/tiff"},
                "thumbnail": {"href": "https://example.com/thumbnail.jpg", "type": "image/jpeg"},
                "metadata": {"href": "https://example.com/metadata.xml", "type": "application/xml"},
                "alternate_url": {
                    "href": "https://example.com/standard.tif",
                    "alternate": {"download": {"href": "https://example.com/alternate/download.tif"}},
                },
            },
        }
        self.pystac_item = pystac.Item.from_dict(self.sample_item_dict)

    def test_init_with_parameters(self):
        """Test ItemDownloader initialization with different parameters"""
        downloader1 = ItemDownloader()
        self.assertEqual(downloader1.max_workers, 5)
        self.assertEqual(downloader1.timeout, 120)
        self.assertTrue(downloader1.allow_redirects)

        downloader2 = ItemDownloader(max_workers=10, timeout=30, allow_redirects=False)
        self.assertEqual(downloader2.max_workers, 10)
        self.assertEqual(downloader2.timeout, 30)
        self.assertFalse(downloader2.allow_redirects)

    def test_get_asset_href_regular(self):
        """Test _get_asset_href with regular href"""
        asset = {"href": "https://example.com/file.tif"}
        href = self.downloader._get_asset_href(asset, "href")
        self.assertEqual(href, "https://example.com/file.tif")

    def test_get_asset_href_nested(self):
        """Test _get_asset_href with nested href"""
        asset = {
            "href": "https://example.com/file.tif",
            "alternate": {"download": {"href": "https://example.com/alternate/file.tif"}},
        }
        href = self.downloader._get_asset_href(asset, "alternate.download.href")
        self.assertEqual(href, "https://example.com/alternate/file.tif")

    def test_get_asset_href_missing(self):
        """Test _get_asset_href with missing href"""
        asset = {"title": "No href here"}
        href = self.downloader._get_asset_href(asset, "href")
        self.assertIsNone(href)

        asset = {"alternate": {"preview": {"href": "https://example.com/file.jpg"}}}
        href = self.downloader._get_asset_href(asset, "alternate.download.href")
        self.assertIsNone(href)

    def test_get_asset_href_default(self):
        """Test _get_asset_href with None href_type (should use default href)"""
        asset = {
            "href": "https://example.com/file.tif",
            "alternate": {"download": {"href": "https://example.com/alternate/file.tif"}},
        }
        href = self.downloader._get_asset_href(asset, None)
        self.assertEqual(href, "https://example.com/file.tif")

    @patch("earthdaily.platform._stac_item_downloader.get_resolver_for_url")
    @patch("earthdaily.platform._stac_item_downloader.CustomHeadersDownloader")
    def test_download_single_asset_success(self, mock_custom_downloader_cls, mock_get_resolver):
        """Test _download_single_asset with successful download"""
        mock_resolver = Mock()
        mock_resolver.get_download_url.return_value = "https://example.com/download/file.tif"
        mock_resolver.get_headers.return_value = {"Authorization": "Bearer token"}
        mock_get_resolver.return_value = mock_resolver

        mock_downloader = Mock()
        mock_custom_downloader_cls.return_value = mock_downloader

        mock_file_path = Path("/test/output/file.tif")
        mock_downloader.download_file.return_value = ("https://example.com/file.tif", mock_file_path)

        output_path = Path("/test/output")
        result = self.downloader._download_single_asset(
            url="https://example.com/file.tif",
            headers={"Authorization": "Bearer token"},
            output_path=output_path,
            quiet=False,
            continue_on_error=False,
        )

        self.assertEqual(result, mock_file_path)
        mock_downloader.download_file.assert_called_once_with(
            file_url="https://example.com/file.tif", save_location=output_path, quiet=False, continue_on_error=False
        )

    @patch("earthdaily.platform._stac_item_downloader.get_resolver_for_url")
    @patch("earthdaily.platform._stac_item_downloader.CustomHeadersDownloader")
    def test_download_single_asset_error_with_continue(self, mock_custom_downloader_cls, mock_get_resolver):
        """Test _download_single_asset with error and continue_on_error=True"""
        mock_resolver = Mock()
        mock_resolver.get_download_url.return_value = "https://example.com/download/file.tif"
        mock_resolver.get_headers.return_value = {"Authorization": "Bearer token"}
        mock_get_resolver.return_value = mock_resolver

        mock_downloader = Mock()
        mock_custom_downloader_cls.return_value = mock_downloader
        mock_downloader.download_file.side_effect = Exception("Download failed")

        output_path = Path("/test/output")
        result = self.downloader._download_single_asset(
            url="https://example.com/file.tif",
            headers={"Authorization": "Bearer token"},
            output_path=output_path,
            quiet=False,
            continue_on_error=True,
        )

        self.assertIsNone(result)

    @patch("earthdaily.platform._stac_item_downloader.get_resolver_for_url")
    @patch("earthdaily.platform._stac_item_downloader.CustomHeadersDownloader")
    def test_download_single_asset_error_without_continue(self, mock_custom_downloader_cls, mock_get_resolver):
        """Test _download_single_asset with error and continue_on_error=False"""
        mock_resolver = Mock()
        mock_resolver.get_download_url.return_value = "https://example.com/download/file.tif"
        mock_resolver.get_headers.return_value = {"Authorization": "Bearer token"}
        mock_get_resolver.return_value = mock_resolver

        mock_downloader = Mock()
        mock_custom_downloader_cls.return_value = mock_downloader
        mock_downloader.download_file.side_effect = Exception("Download failed")

        output_path = Path("/test/output")
        with self.assertRaises(Exception) as context:
            self.downloader._download_single_asset(
                url="https://example.com/file.tif",
                headers={"Authorization": "Bearer token"},
                output_path=output_path,
                quiet=False,
                continue_on_error=False,
            )

        self.assertEqual(str(context.exception), "Download failed")

    @patch("pathlib.Path.mkdir")
    @patch("earthdaily.platform._stac_item_downloader.get_resolver_for_url")
    @patch("earthdaily.platform._stac_item_downloader.ThreadPoolExecutor")
    def test_download_assets_dict_with_thread_pool(self, mock_executor_cls, mock_get_resolver, mock_mkdir):
        """Test download_assets with dict and concurrent downloads"""
        mock_executor = Mock()
        mock_executor_cls.return_value.__enter__.return_value = mock_executor

        mock_future1 = Mock()
        mock_future1.result.return_value = Path("/test/output/visual.tif")
        mock_future2 = Mock()
        mock_future2.result.return_value = Path("/test/output/thumbnail.jpg")

        mock_executor.submit.side_effect = [mock_future1, mock_future2]

        with patch("earthdaily.platform._stac_item_downloader.as_completed", return_value=[mock_future1, mock_future2]):
            result = self.downloader.download_assets(
                item=self.sample_item_dict, asset_keys=["visual", "thumbnail"], output_dir="/test/output"
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(result["visual"], Path("/test/output/visual.tif"))
        self.assertEqual(result["thumbnail"], Path("/test/output/thumbnail.jpg"))

    @patch("pathlib.Path.mkdir")
    @patch("earthdaily.platform._stac_item_downloader.ItemDownloader._download_single_asset")
    def test_download_assets_sequential(self, mock_download_single, mock_mkdir):
        """Test download_assets with sequential downloads (max_workers=1)"""
        sequential_downloader = ItemDownloader(max_workers=1)

        mock_download_single.side_effect = [Path("/test/output/visual.tif"), Path("/test/output/thumbnail.jpg")]

        result = sequential_downloader.download_assets(
            item=self.sample_item_dict, asset_keys=["visual", "thumbnail"], output_dir="/test/output"
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result["visual"], Path("/test/output/visual.tif"))
        self.assertEqual(result["thumbnail"], Path("/test/output/thumbnail.jpg"))
        self.assertEqual(mock_download_single.call_count, 2)

    @patch("pathlib.Path.mkdir")
    def test_download_assets_pystac_item(self, mock_mkdir):
        """Test download_assets with a PySTAC Item"""
        with patch.object(self.downloader, "_download_single_asset") as mock_download_single:
            mock_download_single.return_value = Path("/test/output/visual.tif")

            result = self.downloader.download_assets(
                item=self.pystac_item, asset_keys=["visual"], output_dir="/test/output"
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result["visual"], Path("/test/output/visual.tif"))

    @patch("pathlib.Path.mkdir")
    def test_download_assets_with_href_type(self, mock_mkdir):
        """Test download_assets with custom href_type"""
        with patch.object(self.downloader, "_download_single_asset") as mock_download_single:
            mock_download_single.return_value = Path("/test/output/alternate.tif")

            result = self.downloader.download_assets(
                item=self.sample_item_dict,
                asset_keys=["alternate_url"],
                output_dir="/test/output",
                href_type="alternate.download.href",
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result["alternate_url"], Path("/test/output/alternate.tif"))

    @patch("pathlib.Path.mkdir")
    def test_download_assets_invalid_asset_key_with_continue(self, mock_mkdir):
        """Test download_assets with invalid asset key and continue_on_error=True"""
        with patch.object(self.downloader, "_download_single_asset") as mock_download_single:
            mock_download_single.return_value = Path("/test/output/visual.tif")

            result = self.downloader.download_assets(
                item=self.sample_item_dict,
                asset_keys=["visual", "nonexistent"],
                output_dir="/test/output",
                continue_on_error=True,
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result["visual"], Path("/test/output/visual.tif"))

    def test_download_assets_invalid_asset_key_without_continue(self):
        """Test download_assets with invalid asset key and continue_on_error=False"""
        with self.assertRaises(ValueError) as context:
            self.downloader.download_assets(
                item=self.sample_item_dict,
                asset_keys=["visual", "nonexistent"],
                output_dir="/test/output",
                continue_on_error=False,
            )

        self.assertIn("Asset key 'nonexistent' not found in item", str(context.exception))

    def test_download_assets_invalid_item_type(self):
        """Test download_assets with invalid item type"""
        with self.assertRaises(TypeError) as context:
            self.downloader.download_assets(item="not a dict or pystac.Item", output_dir="/test/output")

        self.assertIn("Item must be a PySTAC Item or a dictionary", str(context.exception))

    def test_download_assets_invalid_pystac_dict(self):
        """Test download_assets with an invalid STAC Item structure"""
        invalid_item = {"type": "NotFeature", "id": "test"}

        with patch("pystac.Item.from_dict", side_effect=Exception("Invalid STAC Item")):
            with self.assertRaises(ValueError) as context:
                self.downloader.download_assets(item=invalid_item, output_dir="/test/output")

        self.assertIn("Provided item is not a valid PySTAC Item or dictionary", str(context.exception))


class TestCustomHeadersDownloader(unittest.TestCase):
    def setUp(self):
        self.custom_headers = {"Authorization": "Bearer token", "X-Custom-Header": "Value"}
        self.downloader = CustomHeadersDownloader(
            supported_protocols=["http", "https"], allow_redirects=True, custom_headers=self.custom_headers
        )

    def test_init(self):
        """Test initialization with custom headers"""
        self.assertEqual(self.downloader.custom_headers, self.custom_headers)
        self.assertEqual(self.downloader.supported_protocols, ["http", "https"])
        self.assertTrue(self.downloader.allow_redirects)

        downloader = CustomHeadersDownloader(supported_protocols=["http"])
        self.assertEqual(downloader.custom_headers, {})

    def test_get_request_headers(self):
        """Test that get_request_headers includes the custom headers"""
        base_headers = {"User-Agent": "earthdaily-python-client"}

        with patch(
            "earthdaily.platform._stac_item_downloader.HttpDownloader.get_request_headers",
            return_value=base_headers.copy(),
        ):
            headers = self.downloader.get_request_headers()

            self.assertIn("User-Agent", headers)
            self.assertIn("Authorization", headers)
            self.assertIn("X-Custom-Header", headers)
            self.assertEqual(headers["Authorization"], "Bearer token")
            self.assertEqual(headers["X-Custom-Header"], "Value")


if __name__ == "__main__":
    unittest.main()
