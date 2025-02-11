import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from earthdaily._uploader import HttpUploader
from earthdaily.exceptions import UnsupportedAssetException


class TestHttpUploader(unittest.TestCase):
    def setUp(self):
        self.uploader = HttpUploader()

    def test_init_with_custom_params(self):
        custom_uploader = HttpUploader(
            supported_protocols=["ftp"],
            timeout=60,
            chunk_size=4096,
            quiet=True,
            continue_on_error=True,
        )
        self.assertEqual(custom_uploader.supported_protocols, ["ftp"])
        self.assertEqual(custom_uploader.timeout, 60)
        self.assertEqual(custom_uploader.chunk_size, 4096)
        self.assertTrue(custom_uploader.quiet)
        self.assertTrue(custom_uploader.continue_on_error)

    def test_check_protocol_valid(self):
        try:
            self.uploader._check_protocol("https://example.com")
        except UnsupportedAssetException:
            self.fail("_check_protocol raised UnsupportedAssetException unexpectedly")

    def test_check_protocol_invalid(self):
        with self.assertRaises(UnsupportedAssetException):
            self.uploader._check_protocol("ftp://example.com")

    @patch("requests.put")
    def test_upload_file_success_non_chunked(self, mock_put):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"Test content")
            temp_file_path = temp_file.name

        try:
            self.uploader.upload_file(temp_file_path, "https://example.com/upload")
            mock_put.assert_called_once()
            # Check that data argument is bytes (non-chunked upload)
            self.assertIsInstance(mock_put.call_args[1]["data"], bytes)
        finally:
            os.unlink(temp_file_path)

    @patch("requests.put")
    def test_upload_file_http_error(self, mock_put):
        mock_put.side_effect = requests.exceptions.HTTPError("404 Client Error")

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"Test content")
            temp_file_path = temp_file.name

        try:
            with self.assertRaises(requests.exceptions.HTTPError):
                self.uploader.upload_file(temp_file_path, "https://example.com/upload")
        finally:
            os.unlink(temp_file_path)

    def test_upload_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.uploader.upload_file("/path/to/nonexistent/file.txt", "https://example.com/upload")


if __name__ == "__main__":
    unittest.main()
