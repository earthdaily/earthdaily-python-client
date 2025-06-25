import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from requests.exceptions import ConnectionError, HTTPError

from earthdaily._downloader import HttpDownloader
from earthdaily.exceptions import UnsupportedAssetException


class TestHttpDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = HttpDownloader(supported_protocols=["http", "https"])
        self.temp_dir = tempfile.mkdtemp()
        self.save_location = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        downloader = HttpDownloader(supported_protocols=["http", "https"])
        self.assertEqual(downloader.supported_protocols, ["http", "https"])
        self.assertTrue(downloader.allow_redirects)

        downloader = HttpDownloader(supported_protocols=["http", "https"], allow_redirects=False)
        self.assertEqual(downloader.supported_protocols, ["http", "https"])
        self.assertFalse(downloader.allow_redirects)

    def test_is_supported_file(self):
        self.assertTrue(self.downloader.is_supported_file("http://example.com/file.txt"))
        self.assertTrue(self.downloader.is_supported_file("https://example.com/file.txt"))

        self.assertFalse(self.downloader.is_supported_file("ftp://example.com/file.txt"))
        self.assertFalse(self.downloader.is_supported_file("file:///path/to/file.txt"))

        custom_downloader = HttpDownloader(supported_protocols=["ftp"])
        self.assertTrue(custom_downloader.is_supported_file("ftp://example.com/file.txt"))
        self.assertFalse(custom_downloader.is_supported_file("http://example.com/file.txt"))

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.url = "http://example.com/file.txt"
        mock_response.iter_content.return_value = [b"content"]
        mock_get.return_value.__enter__.return_value = mock_response

        result = self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)

        self.assertEqual(result[0], "http://example.com/file.txt")
        self.assertEqual(result[1], self.save_location / "file.txt")
        mock_get.assert_called_once()

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_redirect(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.url = "http://example.com/redirected-file.txt"
        mock_response.iter_content.return_value = [b"content"]
        mock_get.return_value.__enter__.return_value = mock_response

        result = self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)

        self.assertEqual(result[0], "http://example.com/file.txt")
        self.assertEqual(result[1], self.save_location / "redirected-file.txt")

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_redirect_edge_cases(self, mock_get):
        """Test edge cases in URL redirection handling"""

        # Case 1: Redirected URL with filename that has no extension
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.url = "http://example.com/redirected-file-without-extension"
        mock_response.iter_content.return_value = [b"content"]
        mock_get.return_value.__enter__.return_value = mock_response

        result = self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)

        self.assertEqual(result[1], self.save_location / "file.txt")

        # Case 2: Redirected URL with empty filename
        mock_response.url = "http://example.com/"

        result = self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)

        self.assertEqual(result[1], self.save_location / "file.txt")

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_with_redirects_disabled(self, mock_get):
        """Test behavior when redirects are disabled but server redirects anyway"""
        no_redirect_downloader = HttpDownloader(supported_protocols=["http", "https"], allow_redirects=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.url = "http://example.com/redirected-file.txt"
        mock_response.iter_content.return_value = [b"content"]
        mock_get.return_value.__enter__.return_value = mock_response

        result = no_redirect_downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)

        self.assertEqual(result[0], "http://example.com/file.txt")
        self.assertEqual(result[1], self.save_location / "file.txt")

        mock_get.assert_called_with(
            "http://example.com/file.txt", stream=True, allow_redirects=False, headers={}, timeout=120
        )

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_unsupported_protocol(self, mock_get):
        with self.assertRaises(UnsupportedAssetException):
            self.downloader.download_file("ftp://example.com/file.txt", self.save_location)
        mock_get.assert_not_called()

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Client Error")
        mock_get.return_value.__enter__.return_value = mock_response

        with self.assertRaises(HTTPError):
            self.downloader.download_file("http://example.com/file.txt", self.save_location, continue_on_error=False)

        result = self.downloader.download_file(
            "http://example.com/file.txt", self.save_location, continue_on_error=True
        )
        self.assertIsNone(result)

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_connection_error(self, mock_get):
        mock_get.return_value.__enter__.side_effect = ConnectionError("Connection refused")

        with self.assertRaises(ConnectionError):
            self.downloader.download_file("http://example.com/file.txt", self.save_location, continue_on_error=False)

        result = self.downloader.download_file(
            "http://example.com/file.txt", self.save_location, continue_on_error=True
        )
        self.assertIsNone(result)

    @patch("earthdaily._downloader.tqdm")
    @patch("earthdaily._downloader.requests.get")
    def test_download_file_quiet_mode(self, mock_get, mock_tqdm):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.url = "http://example.com/file.txt"
        mock_response.iter_content.return_value = [b"content"]
        mock_get.return_value.__enter__.return_value = mock_response

        self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)
        mock_tqdm.assert_not_called()

        self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=False)
        mock_tqdm.assert_called()

    def test_get_request_headers(self):
        # Test default implementation returns empty dict
        headers = self.downloader.get_request_headers()
        self.assertEqual(headers, {})


if __name__ == "__main__":
    unittest.main()
