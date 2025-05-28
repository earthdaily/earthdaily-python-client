from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from earthdaily._eds_models import BaseRequest, BaseResponse
from earthdaily._http_client import HTTPRequest
from earthdaily._uploader import HttpUploader
from earthdaily._zipper import Zipper
from earthdaily.exceptions import EDSAPIError
from earthdaily.platform.models import (
    UploadConfig,
)


@dataclass
class BulkDeleteRequest(BaseRequest):
    """
    Request model for bulk delete operations in the Earth Data Store (EDS) platform.

    Attributes:
    -----------
    upload_config: Dict[str, str]
        Configuration for the upload process.
    """

    upload_config: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "upload_config": self.upload_config,
        }


@dataclass
class ItemError:
    """
    Represents an error associated with a specific item in a bulk operation.

    Attributes:
    -----------
    item_id: str
        The identifier of the item that encountered an error.
    message: str
        A description of the error.
    """

    item_id: str
    message: str


@dataclass
class BulkDeleteJob(BaseResponse):
    """
    Represents the status and details of a bulk delete job.

    Attributes:
    -----------
    job_id: str
        The unique identifier for the job.
    status: str
        The current status of the job.
    message: str
        A message describing the current state or result of the job.
    items_total_count: int
        The total number of items in the job.
    items_error_count: int
        The number of items that encountered errors.
    items_skipped_count: int
        The number of items that were skipped.
    item_errors_subset: List[ItemError]
        A subset of item errors encountered during the job.
    items_skipped_subset: List[ItemError]
        A subset of items that were skipped during the job.
    job_type: str
        The type of the job, e.g., "DELETE".
    items_deleted_count: int
        The number of items successfully deleted.
    upload_config: Optional[UploadConfig]
        The configuration used for uploading data, if applicable.
    """

    job_id: str
    status: str
    message: str
    items_total_count: int
    items_error_count: int
    items_skipped_count: int
    item_errors_subset: List[ItemError]
    items_skipped_subset: List[ItemError]
    job_type: str
    items_deleted_count: int
    upload_config: Optional[UploadConfig] = None

    _service: Optional["BulkDeleteService"] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BulkDeleteJob":
        return cls(
            job_id=data["job_id"],
            status=data["status"],
            message=data["message"],
            items_total_count=data["items_total_count"],
            items_error_count=data["items_error_count"],
            items_skipped_count=data["items_skipped_count"],
            item_errors_subset=[ItemError(**item) for item in data.get("item_errors_subset", [])],
            items_skipped_subset=[ItemError(**item) for item in data.get("items_skipped_subset", [])],
            job_type=data["job_type"],
            items_deleted_count=data["items_deleted_count"],
            upload_config=UploadConfig(**data["upload_config"]) if "upload_config" in data else None,
        )

    def upload(self, file_path: Path, uploader: Optional[HttpUploader] = None):
        if self.upload_config is None:
            raise ValueError("Cannot upload file: upload_config is None")
        uploader = uploader or HttpUploader()
        uploader.upload_file(file_path, self.upload_config.presigned_url)

    def start(self):
        """
        Start a bulk delete job.

        Parameters:
        -----------
        job_id : str
            The ID of the bulk delete job to start.

        Returns:
        --------
        BulkDeleteJob
            Object containing the updated status and details of the bulk delete job.
        """

        if self._service:
            return self._service.start(self.job_id)
        else:
            raise ValueError("Service not available for starting the job")


@dataclass
class BulkDeleteService:
    def __init__(self, api_requester):
        """
        Initialize the BulkDeleteService.

        Parameters:
        -----------
        api_requester : APIRequester
            An instance of APIRequester used to send HTTP requests to the EDS API.
        """
        self.api_requester = api_requester

    def _send_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a request to a platform endpoint.

        Parameters:
        -----------
        method : str
            HTTP method for the request (e.g., "GET", "POST").
        endpoint : str
            The specific endpoint to send the request to.
        data : Optional[Dict[str, Any]], optional
            The payload to be sent in the request.

        Returns:
        --------
        Dict[str, Any]
            The response from the platform API.

        Raises:
        -------
        EDSAPIError
            If the API returns an error response.
        """
        url = f"{self.api_requester.base_url}/platform/v1/stac/{endpoint}"
        request = HTTPRequest(method=method, url=url, body=data)
        response = self.api_requester.send_request(request)

        if response.status_code != 200:
            raise EDSAPIError(f"API request failed with status {response.status_code}: {response.body}")

        return response.body

    def zip_dir(self, source: Path) -> Path:
        """
        Zips the contents of the specified directory or file.
        The resultant ZIP file is saved in the same location as the source,
        with a uniquely generated name to prevent overwriting any existing files.

        Parameters:
        ----------
        source : Path
            The path to the directory or file to be zipped.
            This must be a valid path, and the directory or file must exist.

        Returns:
        -------
        Path
            The path to the newly created ZIP file.
        """
        output_path = source.parent / f"{uuid4()}.zip"

        zipper = Zipper()
        return zipper.zip_content(source, output_path)

    def create(self, collection_id: str) -> BulkDeleteJob:
        """
        Create a bulk delete request to the platform.

        Parameters:
        -----------
        collection_id : str
            The ID of the collection to delete items from.

        Returns:
        --------
        BulkDeleteJob
            The response from the platform API, containing job details and upload configuration.
        """
        request_data = BulkDeleteRequest(
            upload_config={"type": "PRESIGNED_URL"},
        )
        response = self._send_request("POST", f"bulk_jobs/delete/{collection_id}", request_data.to_dict())
        job = BulkDeleteJob.from_dict(response)
        job._service = self
        return job

    def fetch(self, job_id: str) -> BulkDeleteJob:
        """
        Get the status of a bulk delete job.

        Parameters:
        -----------
        job_id : str
            The ID of the bulk delete job to check.

        Returns:
        --------
        BulkDeleteJob
            Object containing the status and details of the bulk delete job.
        """
        response = self._send_request("GET", f"bulk_jobs/delete/{job_id}")
        return BulkDeleteJob.from_dict(response)

    def start(self, job_id: str):
        """
        Start a bulk delete job.

        Parameters:
        -----------
        job_id : str
            The ID of the bulk delete job to start.

        Returns:
        --------
        BulkDeleteJob
            Object containing the updated status and details of the bulk delete job.
        """

        self._send_request("GET", f"bulk_jobs/delete/{job_id}/start")
