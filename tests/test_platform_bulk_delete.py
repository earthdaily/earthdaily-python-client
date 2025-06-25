import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from earthdaily.exceptions import EDSAPIError
from earthdaily.platform._bulk_delete import (
    BulkDeleteJob,
    BulkDeleteService,
    ItemError,
    UploadConfig,
)


class TestBulkDeleteService(unittest.TestCase):
    def setUp(self):
        self.mock_api_requester = Mock()
        self.mock_api_requester.base_url = "https://api.example.com"
        self.bulk_delete_service = BulkDeleteService(self.mock_api_requester)

    def test_create_bulk_delete_success(self):
        mock_response = {
            "job_id": "test_job_id",
            "status": "WAITING_UPLOAD",
            "message": "Job created",
            "items_total_count": 0,
            "items_error_count": 0,
            "items_skipped_count": 0,
            "item_errors_subset": [],
            "items_skipped_subset": [],
            "job_type": "DELETE",
            "items_deleted_count": 0,
            "upload_config": {"type": "PRESIGNED_URL", "presigned_url": "https://example.com/upload"},
        }

        self.mock_api_requester.send_request.return_value.status_code = 200
        self.mock_api_requester.send_request.return_value.body = mock_response

        result = self.bulk_delete_service.create(collection_id="test_collection")

        self.assertIsInstance(result, BulkDeleteJob)
        self.assertEqual(result.job_id, "test_job_id")
        self.assertEqual(result.status, "WAITING_UPLOAD")
        self.assertIsNotNone(result.upload_config)
        self.assertEqual(result.upload_config.presigned_url, "https://example.com/upload")

        # Verify the request was made correctly
        expected_url = "https://api.example.com/platform/v1/stac/bulk_jobs/delete/test_collection"
        expected_body = {
            "upload_config": {"type": "PRESIGNED_URL"},
        }
        self.mock_api_requester.send_request.assert_called_once()
        actual_request = self.mock_api_requester.send_request.call_args[0][0]
        self.assertEqual(actual_request.method, "POST")
        self.assertEqual(actual_request.url, expected_url)
        self.assertEqual(actual_request.body, expected_body)

    def test_fetch_job_status(self):
        mock_response = {
            "job_id": "test_job_id",
            "status": "RUNNING",
            "message": "Job in progress",
            "items_total_count": 10,
            "items_error_count": 1,
            "items_skipped_count": 2,
            "item_errors_subset": [{"item_id": "item1", "message": "Error processing item"}],
            "items_skipped_subset": [
                {"item_id": "item2", "message": "Item skipped"},
                {"item_id": "item3", "message": "Another skip"},
            ],
            "job_type": "DELETE",
            "items_deleted_count": 7,
        }

        self.mock_api_requester.send_request.return_value.status_code = 200
        self.mock_api_requester.send_request.return_value.body = mock_response

        result = self.bulk_delete_service.fetch("test_job_id")

        self.assertIsInstance(result, BulkDeleteJob)
        self.assertEqual(result.status, "RUNNING")
        self.assertEqual(result.items_total_count, 10)
        self.assertEqual(result.items_error_count, 1)
        self.assertEqual(result.items_skipped_count, 2)
        self.assertEqual(result.items_deleted_count, 7)
        self.assertEqual(len(result.item_errors_subset), 1)
        self.assertEqual(len(result.items_skipped_subset), 2)
        self.assertIsInstance(result.item_errors_subset[0], ItemError)
        self.assertIsInstance(result.items_skipped_subset[0], ItemError)

        # Verify the request was made correctly
        expected_url = "https://api.example.com/platform/v1/stac/bulk_jobs/delete/test_job_id"
        self.mock_api_requester.send_request.assert_called_once()
        actual_request = self.mock_api_requester.send_request.call_args[0][0]
        self.assertEqual(actual_request.method, "GET")
        self.assertEqual(actual_request.url, expected_url)

    def test_start_job(self):
        # First, test the actual start method call
        self.mock_api_requester.send_request.return_value.status_code = 200
        self.mock_api_requester.send_request.return_value.body = {}

        job_id = "test_job_id"
        self.bulk_delete_service.start(job_id)

        expected_url = f"https://api.example.com/platform/v1/stac/bulk_jobs/delete/{job_id}/start"
        self.mock_api_requester.send_request.assert_called_once()
        actual_request = self.mock_api_requester.send_request.call_args[0][0]
        self.assertEqual(actual_request.method, "GET")
        self.assertEqual(actual_request.url, expected_url)

    def test_start_job_error(self):
        self.mock_api_requester.send_request.return_value.status_code = 400
        self.mock_api_requester.send_request.return_value.body = {"error": "Job already started"}

        with self.assertRaises(EDSAPIError) as context:
            self.bulk_delete_service.start("test_job_id")

        self.assertEqual(str(context.exception), "API request failed with status 400: {'error': 'Job already started'}")

    @patch("pathlib.Path.exists")
    def test_zip_dir_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        source_path = Path("/test/nonexistent")

        with self.assertRaises(FileNotFoundError):
            self.bulk_delete_service.zip_dir(source_path)

    def test_api_error_handling(self):
        self.mock_api_requester.send_request.return_value.status_code = 400
        self.mock_api_requester.send_request.return_value.body = {"error": "Bad Request"}

        with self.assertRaises(EDSAPIError) as context:
            self.bulk_delete_service.create(collection_id="test_collection")

        self.assertIn("API request failed with status 400", str(context.exception))

    @patch("earthdaily.platform._bulk_delete.HttpUploader")
    def test_job_upload(self, mock_uploader_class):
        mock_uploader = Mock()
        mock_uploader_class.return_value = mock_uploader

        job = BulkDeleteJob(
            job_id="test_job_id",
            status="WAITING_UPLOAD",
            message="Job created",
            items_total_count=0,
            items_error_count=0,
            items_skipped_count=0,
            item_errors_subset=[],
            items_skipped_subset=[],
            job_type="DELETE",
            items_deleted_count=0,
            upload_config=UploadConfig(type="PRESIGNED_URL", presigned_url="https://example.com/upload"),
        )

        file_path = Path("/test/file.zip")
        job.upload(file_path)

        mock_uploader.upload_file.assert_called_once_with(file_path, "https://example.com/upload")

    def test_job_start_with_service(self):
        mock_response = {
            "job_id": "test_job_id",
            "status": "RUNNING",
            "message": "Job started",
            "items_total_count": 0,
            "items_error_count": 0,
            "items_skipped_count": 0,
            "item_errors_subset": [],
            "items_skipped_subset": [],
            "job_type": "DELETE",
            "items_deleted_count": 0,
        }

        self.mock_api_requester.send_request.return_value.status_code = 200
        self.mock_api_requester.send_request.return_value.body = mock_response

        job = BulkDeleteJob(
            job_id="test_job_id",
            status="WAITING_UPLOAD",
            message="Job created",
            items_total_count=0,
            items_error_count=0,
            items_skipped_count=0,
            item_errors_subset=[],
            items_skipped_subset=[],
            job_type="DELETE",
            items_deleted_count=0,
            _service=self.bulk_delete_service,
        )

        result = job.start()
        self.assertIsNone(result)

    def test_job_start_without_service(self):
        job = BulkDeleteJob(
            job_id="test_job_id",
            status="WAITING_UPLOAD",
            message="Job created",
            items_total_count=0,
            items_error_count=0,
            items_skipped_count=0,
            item_errors_subset=[],
            items_skipped_subset=[],
            job_type="DELETE",
            items_deleted_count=0,
        )

        with self.assertRaises(ValueError):
            job.start()


if __name__ == "__main__":
    unittest.main()
