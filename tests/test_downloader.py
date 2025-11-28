import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from requests.exceptions import ConnectionError, HTTPError

from earthdaily._downloader import HttpDownloader
from earthdaily.exceptions import DownloadValidationError, UnsupportedAssetException


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
        content = b"content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.url = "http://example.com/file.txt"
        mock_response.iter_content.return_value = [content]
        mock_get.return_value.__enter__.return_value = mock_response

        result = self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)

        self.assertEqual(result[0], "http://example.com/file.txt")
        self.assertEqual(result[1], self.save_location / "file.txt")
        mock_get.assert_called_once()

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_redirect(self, mock_get):
        content = b"content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.url = "http://example.com/redirected-file.txt"
        mock_response.iter_content.return_value = [content]
        mock_get.return_value.__enter__.return_value = mock_response

        result = self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)

        self.assertEqual(result[0], "http://example.com/file.txt")
        self.assertEqual(result[1], self.save_location / "redirected-file.txt")

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_redirect_edge_cases(self, mock_get):
        content = b"content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.url = "http://example.com/redirected-file-without-extension"
        mock_response.iter_content.return_value = [content]
        mock_get.return_value.__enter__.return_value = mock_response

        result = self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)

        self.assertEqual(result[1], self.save_location / "file.txt")

        mock_response.url = "http://example.com/"

        result = self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)

        self.assertEqual(result[1], self.save_location / "file.txt")

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_with_redirects_disabled(self, mock_get):
        no_redirect_downloader = HttpDownloader(supported_protocols=["http", "https"], allow_redirects=False)

        content = b"content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.url = "http://example.com/redirected-file.txt"
        mock_response.iter_content.return_value = [content]
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
        content = b"content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.url = "http://example.com/file.txt"
        mock_response.iter_content.return_value = [content]
        mock_get.return_value.__enter__.return_value = mock_response

        self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)
        mock_tqdm.assert_not_called()

        self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=False)
        mock_tqdm.assert_called()

    def test_get_request_headers(self):
        headers = self.downloader.get_request_headers()
        self.assertEqual(headers, {})

    def test_validate_chunk_size_valid(self):
        self.downloader._validate_chunk_size(1)
        self.downloader._validate_chunk_size(8192)
        self.downloader._validate_chunk_size(100 * 1024 * 1024)

    def test_validate_chunk_size_too_small(self):
        with self.assertRaises(DownloadValidationError) as context:
            self.downloader._validate_chunk_size(0)
        self.assertIn("must be at least", str(context.exception))

        with self.assertRaises(DownloadValidationError) as context:
            self.downloader._validate_chunk_size(-1)
        self.assertIn("must be at least", str(context.exception))

    def test_validate_chunk_size_too_large(self):
        with self.assertRaises(DownloadValidationError) as context:
            self.downloader._validate_chunk_size(100 * 1024 * 1024 + 1)
        self.assertIn("must not exceed", str(context.exception))

    def test_validate_chunk_size_non_integer(self):
        with self.assertRaises(DownloadValidationError) as context:
            self.downloader._validate_chunk_size(8192.5)
        self.assertIn("must be an integer", str(context.exception))

        with self.assertRaises(DownloadValidationError) as context:
            self.downloader._validate_chunk_size("8192")
        self.assertIn("must be an integer", str(context.exception))

    def test_validate_total_size_valid(self):
        self.downloader._validate_total_size(100, 100, "http://example.com/file.txt")
        self.downloader._validate_total_size(0, 100, "http://example.com/file.txt")

    def test_validate_total_size_mismatch(self):
        with self.assertRaises(DownloadValidationError) as context:
            self.downloader._validate_total_size(100, 50, "http://example.com/file.txt")
        self.assertIn("size mismatch", str(context.exception))
        self.assertIn("expected 100 bytes", str(context.exception))
        self.assertIn("got 50 bytes", str(context.exception))

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_invalid_chunk_size(self, mock_get):
        with self.assertRaises(DownloadValidationError):
            self.downloader.download_file("http://example.com/file.txt", self.save_location, chunk_size=0)
        mock_get.assert_not_called()

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_invalid_chunk_size_continue_on_error(self, mock_get):
        result = self.downloader.download_file(
            "http://example.com/file.txt", self.save_location, chunk_size=-1, continue_on_error=True
        )
        self.assertIsNone(result)

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_size_mismatch(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.url = "http://example.com/file.txt"
        mock_response.iter_content.return_value = [b"short"]
        mock_get.return_value.__enter__.return_value = mock_response

        with self.assertRaises(DownloadValidationError) as context:
            self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)
        self.assertIn("size mismatch", str(context.exception))

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_size_mismatch_continue_on_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.url = "http://example.com/file.txt"
        mock_response.iter_content.return_value = [b"short"]
        mock_get.return_value.__enter__.return_value = mock_response

        result = self.downloader.download_file(
            "http://example.com/file.txt", self.save_location, quiet=True, continue_on_error=True
        )
        self.assertIsNone(result)

    @patch("earthdaily._downloader.requests.get")
    def test_download_file_no_content_length_header(self, mock_get):
        content = b"content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.url = "http://example.com/file.txt"
        mock_response.iter_content.return_value = [content]
        mock_get.return_value.__enter__.return_value = mock_response

        result = self.downloader.download_file("http://example.com/file.txt", self.save_location, quiet=True)

        self.assertEqual(result[0], "http://example.com/file.txt")
        self.assertEqual(result[1], self.save_location / "file.txt")


if __name__ == "__main__":
    unittest.main()
