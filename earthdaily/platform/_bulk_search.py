from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from earthdaily._downloader import HttpDownloader
from earthdaily._eds_models import BaseRequest, BaseResponse
from earthdaily._http_client import HTTPRequest
from earthdaily.exceptions import EDSAPIError


@dataclass
class BulkSearchRequest(BaseRequest):
    """
    Request model for bulk read operations in the Earth Data Store (EDS) platform.

    This class inherits from BaseRequest and includes all parameters
    necessary for creating a bulk read request.

    Attributes:
    -----------
    collections: Optional[List[Union[str, Any]]]
        List of one or more Collection IDs or pystac.Collection instances.
    export_format: Optional[str]
        The format to export the results in.
    limit: int
        A recommendation to the service as to the number of items to return
        per page of results. Defaults to 100.
    ids: Optional[List[str]]
        List of one or more Item ids to filter on.
    bbox: Optional[Union[List[float], tuple, Any]]
        A list, tuple, or iterator representing a bounding box of 2D
        or 3D coordinates. Results will be filtered to only those
        intersecting the bounding box.
    intersects: Optional[Union[str, Dict[str, Any]]]
        A string or dictionary representing a GeoJSON geometry or feature,
        or an object that implements a __geo_interface__ property.
    datetime: Optional[str]
        Either a single datetime or datetime range used to filter results.
        Can be a datetime string, a datetime object, or a range string.
    query: Optional[Union[List[Any], Dict[str, Any]]]
        List or JSON of query parameters as per the STAC API query extension.
    filter: Optional[Dict[str, Any]]
        JSON of query parameters as per the STAC API filter extension.
    filter_lang: Optional[str]
        Language variant used in the filter body.
    sortby: Optional[Union[str, List[str]]]
        A single field or list of fields to sort the response by.
    fields: Optional[List[str]]
        A list of fields to include in the response.
    """

    collections: Optional[List[Union[str, Any]]] = None
    export_format: Optional[str] = None
    limit: int = 100
    ids: Optional[List[str]] = None
    bbox: Optional[Union[List[float], tuple, Any]] = None
    intersects: Optional[Union[str, Dict[str, Any]]] = None
    datetime: Optional[str] = None
    query: Optional[Union[List[Any], Dict[str, Any]]] = None
    filter: Optional[Dict[str, Any]] = None
    filter_lang: Optional[str] = None
    sortby: Optional[Union[str, List[str]]] = None
    fields: Optional[List[str]] = None


@dataclass
class BulkSearchCreateResponse(BaseResponse):
    """
    Response model for a successfully created bulk search request.

    This class encapsulates the response received when initiating a bulk search
    operation, containing the unique identifier for tracking the search job.

    Attributes
    ----------
    job_id : str
        A unique identifier for the bulk search job.
    """

    job_id: str


@dataclass
class BulkSearchJob(BaseResponse):
    """
    Represents the current state of a bulk search job.

    This class provides information about a bulk search job's status and available
    assets once completed.

    Attributes
    ----------
    job_id : str
        The unique identifier for the job.
    status : str
        The current status of the job (e.g., "PENDING", "RUNNING", "COMPLETED", "FAILED").
    assets : List[str], default=empty list
        URLs of assets available for download when the job is completed.

    Methods
    -------
    download_assets(save_location: Path, downloader: Optional[HttpDownloader] = None) -> None
        Downloads all available assets to the specified location.
    """

    job_id: str
    status: str
    assets: List[str] = field(default_factory=list)

    def download_assets(self, save_location: Path, downloader: Optional[HttpDownloader] = None) -> None:
        """
        Downloads all assets associated with a completed job.

        Parameters
        ----------
        save_location : Path
            The directory where downloaded assets will be saved.
        downloader : Optional[HttpDownloader], default=None
            A custom downloader instance. If None, creates a new HttpDownloader.

        Raises
        ------
        ValueError
            If the job is not completed or no assets are available.
        """
        if self.status != "COMPLETED":
            raise ValueError(f"Job {self.job_id} is not completed. Current status: {self.status}")

        if not self.assets:
            raise ValueError(f"No assets available for job {self.job_id}")

        downloader = downloader or HttpDownloader(supported_protocols=["http", "https"])

        for asset in self.assets:
            downloader.download_file(file_url=asset, save_location=save_location)


class BulkSearchService:
    """
    Service for managing bulk search operations in the Earth Data Store.

    This class provides methods to create, monitor, and download results from
    bulk search operations.

    Methods
    -------
    create(**kwargs)
        Initiates a new bulk search request.
    fetch(job_id: str)
        Retrieves the current status of a bulk search job.
    download_bulk_read_assets(job_status: BulkSearchJob, save_location: Path)
        Downloads assets from a completed bulk search job.
    """

    def __init__(self, api_requester):
        """
        Initialize the BulkReadService.

        Parameters:
        -----------
        api_requester : APIRequester
            An instance of APIRequester used to send HTTP requests to the EDS API.
        """
        self.api_requester = api_requester

    def create(
        self,
        collections: Optional[List[Union[str, Any]]] = None,
        export_format: Optional[str] = None,
        limit: int = 100,
        ids: Optional[List[str]] = None,
        bbox: Optional[Union[List[float], tuple, Any]] = None,
        intersects: Optional[Union[str, Dict[str, Any]]] = None,
        datetime: Optional[str] = None,
        query: Optional[Union[List[Any], Dict[str, Any]]] = None,
        filter: Optional[Dict[str, Any]] = None,
        filter_lang: Optional[str] = None,
        sortby: Optional[Union[str, List[str]]] = None,
        fields: Optional[List[str]] = None,
    ) -> BulkSearchCreateResponse:
        """
        Create a new bulk search request in the Earth Data Store.

        This method initiates a new asynchronous bulk search operation based on the
        provided parameters. It returns a response containing a job ID that can be
        used to track the progress of the search and retrieve results.

        Parameters:
        -----------
        collections: Optional[List[Union[str, Any]]]
            List of one or more Collection IDs or pystac.Collection instances.
        export_format: Optional[str]
            The format to export the results in.
        limit: int
            A recommendation to the service as to the number of items to return
            per page of results. Defaults to 100.
        ids: Optional[List[str]]
            List of one or more Item ids to filter on.
        bbox: Optional[Union[List[float], tuple, Any]]
            A list, tuple, or iterator representing a bounding box of 2D
            or 3D coordinates. Results will be filtered to only those
            intersecting the bounding box.
        intersects: Optional[Union[str, Dict[str, Any]]]
            A string or dictionary representing a GeoJSON geometry or feature,
            or an object that implements a __geo_interface__ property.
        datetime: Optional[str]
            Either a single datetime or datetime range used to filter results.
            Can be a datetime string, a datetime object, or a range string.
        query: Optional[Union[List[Any], Dict[str, Any]]]
            List or JSON of query parameters as per the STAC API query extension.
        filter: Optional[Dict[str, Any]]
            JSON of query parameters as per the STAC API filter extension.
        filter_lang: Optional[str]
            Language variant used in the filter body.
        sortby: Optional[Union[str, List[str]]]
            A single field or list of fields to sort the response by.
        fields: Optional[List[str]]
            A list of fields to include in the response.

        Returns
        -------
        BulkSearchCreateResponse
            Object containing the job_id for the created bulk search request.

        Raises
        ------
        EDSAPIError
            If the API returns an error response
        """
        request_data = BulkSearchRequest(
            collections=collections,
            export_format=export_format,
            limit=limit,
            ids=ids,
            bbox=bbox,
            intersects=intersects,
            datetime=datetime,
            query=query,
            filter=filter,
            filter_lang=filter_lang,
            sortby=sortby,
            fields=fields,
        )
        response = self._send_request("POST", "search/bulk", request_data.to_dict())
        return BulkSearchCreateResponse(**response)

    def fetch(self, job_id: str) -> BulkSearchJob:
        """
        Get the status of a bulk read job and optionally download assets if the job is completed.

        Parameters:
        -----------
        job_id : str
            The ID of the bulk read job to check.
        download_assets : bool, optional
            Whether to download the assets once the job is completed. Defaults to False.
        save_location : Optional[Path], optional
            Path to the directory where assets will be saved if download_assets is True. Defaults to None.

        Returns:
        --------
        JobStatusResponse
            Object containing the status of the bulk read job.

        Raises:
        -------
        ValueError
            If download_assets is True and save_location is not provided.
        """
        response = self._send_request("GET", f"search/bulk/jobs/{job_id}")
        return BulkSearchJob(**response)

    def download_bulk_read_assets(self, job_status: BulkSearchJob, save_location: Path) -> None:
        """
        Download assets associated with a completed bulk read job.

        Raises:
        -------
        ValueError
            If the job is not completed or no assets are available.
        """
        if job_status.status != "COMPLETED":
            raise ValueError(f"Job {job_status.job_id} is not completed. Current status: {job_status.status}")

        if not job_status.assets:
            raise ValueError(f"No assets available for job {job_status.job_id}")

        downloader = HttpDownloader(supported_protocols=["http", "https"])
        for asset in job_status.assets:
            downloader.download_file(file_url=asset, save_location=save_location)

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
            raise EDSAPIError(
                f"API request failed with status {response.status_code}: {response.body}",
                status_code=response.status_code,
                body=response.body,
            )

        return response.body
