import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from earthdaily._http_client import HTTPRequest, HTTPResponse
from earthdaily.exceptions import EDSAPIError
from earthdaily.platform import BulkSearchService
from earthdaily.platform._bulk_search import BulkSearchCreateResponse, BulkSearchJob


class TestBulkSearchService(unittest.TestCase):
    def setUp(self):
        self.mock_api_requester = Mock()
        self.mock_api_requester.base_url = "https://api.example.com"
        self.service = BulkSearchService(self.mock_api_requester)

    def test_create_success_minimal(self):
        # Prepare mock response
        mock_response = HTTPResponse(200, {"job_id": "test_job_123"}, {})
        self.mock_api_requester.send_request.return_value = mock_response

        # Call the method with minimal parameters
        result = self.service.create(collections=["test_collection"])

        # Assert the result
        self.assertIsInstance(result, BulkSearchCreateResponse)
        self.assertEqual(result.job_id, "test_job_123")

        # Assert the API request
        self.mock_api_requester.send_request.assert_called_once()
        args, _ = self.mock_api_requester.send_request.call_args
        self.assertEqual(args[0].method, "POST")
        self.assertEqual(args[0].url, "https://api.example.com/platform/v1/stac/search/bulk")

        # Assert the request body
        expected_body = {
            "collections": ["test_collection"],
            "limit": 100,  # default limit
        }
        self.assertEqual(args[0].body, expected_body)

    def test_create_success_full_parameters(self):
        mock_response = HTTPResponse(200, {"job_id": "test_job_456"}, {})
        self.mock_api_requester.send_request.return_value = mock_response

        # Call with all possible parameters
        result = self.service.create(
            collections=["test_collection"],
            export_format="geoparquet",
            export_type="standard",
            limit=500,
            ids=["item1", "item2"],
            bbox=[0, 0, 1, 1],
            intersects={"type": "Point", "coordinates": [0, 0]},
            datetime="2023-01-01/2023-12-31",
            query={"eo:cloud_cover": {"lt": 10}},
            filter={"op": "=", "args": [{"property": "collection"}, "sentinel-2"]},
            filter_lang="cql-json",
            sortby=["-datetime", "cloud_cover"],
            fields=["id", "properties.datetime"],
        )

        self.assertIsInstance(result, BulkSearchCreateResponse)
        self.assertEqual(result.job_id, "test_job_456")

        # Verify all parameters were passed correctly
        args, _ = self.mock_api_requester.send_request.call_args
        body = args[0].body
        self.assertEqual(body["collections"], ["test_collection"])
        self.assertEqual(body["export_format"], "geoparquet")
        self.assertEqual(body["export_type"], "standard")
        self.assertEqual(body["limit"], 500)
        self.assertEqual(body["ids"], ["item1", "item2"])
        self.assertEqual(body["bbox"], [0, 0, 1, 1])
        self.assertEqual(body["intersects"], {"type": "Point", "coordinates": [0, 0]})
        self.assertEqual(body["datetime"], "2023-01-01/2023-12-31")
        self.assertEqual(body["query"], {"eo:cloud_cover": {"lt": 10}})
        self.assertEqual(body["filter"], {"op": "=", "args": [{"property": "collection"}, "sentinel-2"]})
        self.assertEqual(body["filter_lang"], "cql-json")
        self.assertEqual(body["sortby"], ["-datetime", "cloud_cover"])
        self.assertEqual(body["fields"], ["id", "properties.datetime"])

    def test_create_error_responses(self):
        error_cases = [
            (400, "Bad Request", "Invalid parameters provided"),
            (401, "Unauthorized", "Authentication required"),
            (403, "Forbidden", "Insufficient permissions"),
            (404, "Not Found", "Resource not found"),
            (500, "Internal Server Error", "Server error occurred"),
        ]

        for status_code, error_msg, expected_error in error_cases:
            with self.subTest(status_code=status_code):
                mock_response = HTTPResponse(status_code, {"error": error_msg}, {})
                self.mock_api_requester.send_request.return_value = mock_response

                with self.assertRaises(EDSAPIError) as context:
                    self.service.create(collections=["test_collection"])

                self.assertIn(str(status_code), str(context.exception))
                self.assertIn(error_msg, str(context.exception))

    def test_fetch_success(self):
        mock_response = HTTPResponse(
            200,
            {
                "job_id": "test_job_123",
                "status": "COMPLETED",
                "assets": ["https://example.com/asset1.tif", "https://example.com/asset2.tif"],
            },
            {},
        )
        self.mock_api_requester.send_request.return_value = mock_response

        result = self.service.fetch("test_job_123")

        self.assertIsInstance(result, BulkSearchJob)
        self.assertEqual(result.job_id, "test_job_123")
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(len(result.assets), 2)

        # Verify API request
        self.mock_api_requester.send_request.assert_called_once()
        args, _ = self.mock_api_requester.send_request.call_args
        self.assertEqual(args[0].method, "GET")
        self.assertEqual(args[0].url, "https://api.example.com/platform/v1/stac/search/bulk/jobs/test_job_123")

    def test_fetch_error_responses(self):
        error_cases = [
            (400, "Bad Request"),
            (404, "Job not found"),
            (500, "Internal Server Error"),
        ]

        for status_code, error_msg in error_cases:
            with self.subTest(status_code=status_code):
                mock_response = HTTPResponse(status_code, {"error": error_msg}, {})
                self.mock_api_requester.send_request.return_value = mock_response

                with self.assertRaises(EDSAPIError) as context:
                    self.service.fetch("test_job_123")

                self.assertIn(str(status_code), str(context.exception))
                self.assertIn(error_msg, str(context.exception))

    def test_bulk_search_job_download_assets_with_provided_downloader(self):
        """Test downloading assets when a downloader is provided."""
        mock_downloader = Mock()

        job = BulkSearchJob(
            job_id="test_job_123",
            status="COMPLETED",
            assets=["https://example.com/asset1.tif", "https://example.com/asset2.tif"],
        )

        save_location = Path("./test_downloads")
        job.download_assets(save_location, downloader=mock_downloader)

        # Verify each asset was downloaded using the provided downloader
        self.assertEqual(mock_downloader.download_file.call_count, 2)
        mock_downloader.download_file.assert_any_call(
            file_url="https://example.com/asset1.tif", save_location=save_location
        )
        mock_downloader.download_file.assert_any_call(
            file_url="https://example.com/asset2.tif", save_location=save_location
        )

    @patch("earthdaily.platform._bulk_search.HttpDownloader")
    def test_bulk_search_job_download_assets_creates_downloader(self, mock_downloader_class):
        """Test downloading assets when no downloader is provided."""
        mock_downloader = Mock()
        mock_downloader_class.return_value = mock_downloader

        job = BulkSearchJob(
            job_id="test_job_123",
            status="COMPLETED",
            assets=["https://example.com/asset1.tif", "https://example.com/asset2.tif"],
        )

        save_location = Path("./test_downloads")
        job.download_assets(save_location)

        # Verify a new downloader was created with correct protocols
        mock_downloader_class.assert_called_once_with(supported_protocols=["http", "https"])

        # Verify each asset was downloaded
        self.assertEqual(mock_downloader.download_file.call_count, 2)
        mock_downloader.download_file.assert_any_call(
            file_url="https://example.com/asset1.tif", save_location=save_location
        )
        mock_downloader.download_file.assert_any_call(
            file_url="https://example.com/asset2.tif", save_location=save_location
        )

    def test_bulk_search_job_download_assets_not_completed(self):
        """Test attempting to download assets when job is not completed."""
        job = BulkSearchJob(job_id="test_job_123", status="RUNNING", assets=[])

        with self.assertRaises(ValueError) as context:
            job.download_assets(Path("./test_downloads"))

        self.assertIn("not completed", str(context.exception))

    def test_bulk_search_job_download_assets_no_assets(self):
        """Test attempting to download when no assets are available."""
        job = BulkSearchJob(job_id="test_job_123", status="COMPLETED", assets=[])

        with self.assertRaises(ValueError) as context:
            job.download_assets(Path("./test_downloads"))

        self.assertIn("No assets available", str(context.exception))

    def test_send_request_custom_endpoint(self):
        mock_response = HTTPResponse(200, {"custom": "data"}, {})
        self.mock_api_requester.send_request.return_value = mock_response

        result = self.service._send_request("GET", "custom/endpoint", {"param": "value"})

        self.assertEqual(result, {"custom": "data"})
        self.mock_api_requester.send_request.assert_called_once()
        args, _ = self.mock_api_requester.send_request.call_args
        self.assertIsInstance(args[0], HTTPRequest)
        self.assertEqual(args[0].method, "GET")
        self.assertEqual(args[0].url, "https://api.example.com/platform/v1/stac/custom/endpoint")
        self.assertEqual(args[0].body, {"param": "value"})


if __name__ == "__main__":
    unittest.main()
