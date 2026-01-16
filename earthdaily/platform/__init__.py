from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Literal, Optional, Union

from pystac import ItemCollection
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from earthdaily._api_requester import APIRequester
from earthdaily._eds_config import AssetAccessMode
from earthdaily._eds_logging import LoggerConfig
from earthdaily.platform._bulk_delete import BulkDeleteService
from earthdaily.platform._bulk_insert import BulkInsertService
from earthdaily.platform._bulk_search import BulkSearchService
from earthdaily.platform._stac_item import StacItemService

logger = LoggerConfig(logger_name=__name__).get_logger()

DatetimeInput = Union[str, tuple[datetime, datetime], tuple[str, str]]


class PlatformService:
    """
    Represents the Platform Service for interacting with specific platform-related endpoints.

    Attributes:
    -----------
    api_requester : APIRequester
        An instance of APIRequester used to send HTTP requests to the EDS API.
    """

    def __init__(self, api_requester: APIRequester, asset_access_mode: AssetAccessMode):
        """
        Initialize the PlatformService.

        Parameters:
        -----------
        api_requester : APIRequester
            An instance of APIRequester used to send HTTP requests to the EDS API.
        asset_access_mode : AssetAccessMode
            The mode of access for assets. Defaults to AssetAccessMode.PRESIGNED_URLS.
        """
        self.api_requester = api_requester
        self.bulk_search = BulkSearchService(api_requester)
        self.bulk_insert = BulkInsertService(api_requester)
        self.bulk_delete = BulkDeleteService(api_requester)
        self.stac_item = StacItemService(api_requester)

        self._stac_url = f"{api_requester.base_url}/platform/v1/stac"
        self._base_stac_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **api_requester.headers,
        }

        self.pystac_client = self._create_pystac_client()

    def _create_pystac_client(self) -> Client:
        """Create a new pystac Client instance with configured retry and fresh auth token."""
        stac_io = StacApiIO()
        retry_strategy = Retry(
            total=self.api_requester.config.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=self.api_requester.config.retry_backoff_factor,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        stac_io.session.mount("http://", adapter)
        stac_io.session.mount("https://", adapter)

        headers = {**self._base_stac_headers}
        if self.api_requester.auth:
            headers["Authorization"] = f"Bearer {self.api_requester.auth.get_token()}"

        return Client.open(
            self._stac_url,
            stac_io=stac_io,
            headers=headers,
        )

    def search(
        self,
        *,
        days_per_chunk: Optional[Union[int, Literal["auto"]]] = None,
        max_workers: int = 10,
        max_items_per_chunk: Optional[int] = None,
        **kwargs: Any,
    ) -> ItemCollection:
        """
        Search STAC items with optional concurrent execution.

        When days_per_chunk is specified, the datetime range is split into chunks
        and searches are executed concurrently using a thread pool.

        Parameters
        ----------
        days_per_chunk : int | "auto" | None, default=None
            If None, performs a standard single search.
            If "auto", automatically calculates chunk size based on date range.
            If int, splits datetime range into chunks of specified days.
        max_workers : int, default=10
            Maximum number of concurrent threads. Capped at 10.
        max_items_per_chunk : int | None, default=None
            Maximum items to fetch per chunk in concurrent search.
            Only used when days_per_chunk is set. If None, fetches all items per chunk.
        **kwargs : Any
            All parameters accepted by pystac_client.Client.search()
            (collections, datetime, bbox, intersects, max_items, query, etc.)

        Returns
        -------
        ItemCollection
            Collection of STAC items matching the search criteria.

        Raises
        ------
        ValueError
            If max_items is used with days_per_chunk. Use max_items_per_chunk instead.
            If days_per_chunk is zero or negative.
            If max_workers is zero or negative.
            If days_per_chunk is used with open-ended datetime intervals (containing '..').
        """
        datetime_param = kwargs.get("datetime")

        if days_per_chunk is None or datetime_param is None:
            return self.pystac_client.search(**kwargs).item_collection()

        if "max_items" in kwargs:
            raise ValueError(
                "max_items cannot be used with concurrent search (days_per_chunk). "
                "Use max_items_per_chunk instead to limit items per chunk."
            )

        if max_workers <= 0:
            raise ValueError(f"max_workers must be a positive integer, got {max_workers}")

        return self._execute_concurrent_search(
            datetime_param=datetime_param,
            days_per_chunk=days_per_chunk,
            max_workers=max_workers,
            max_items_per_chunk=max_items_per_chunk,
            search_kwargs=kwargs,
        )

    def _execute_concurrent_search(
        self,
        datetime_param: DatetimeInput,
        days_per_chunk: Union[int, Literal["auto"]],
        max_workers: int,
        max_items_per_chunk: Optional[int],
        search_kwargs: dict[str, Any],
    ) -> ItemCollection:
        """Execute search concurrently across datetime chunks with thread-safe clients."""
        date_ranges = self._split_datetime(datetime_param, days_per_chunk)

        if len(date_ranges) <= 1:
            if max_items_per_chunk is not None:
                search_kwargs = {**search_kwargs, "max_items": max_items_per_chunk}
            return self.pystac_client.search(**search_kwargs).item_collection()

        effective_workers = min(max_workers, 10, len(date_ranges))
        logger.info(f"Concurrent search: {len(date_ranges)} chunks, {effective_workers} workers")

        kwargs_without_datetime = {k: v for k, v in search_kwargs.items() if k != "datetime"}
        if max_items_per_chunk is not None:
            kwargs_without_datetime["max_items"] = max_items_per_chunk

        def fetch_chunk(dt_range: str) -> list:
            thread_client = self._create_pystac_client()
            return list(thread_client.search(datetime=dt_range, **kwargs_without_datetime).items())

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            results = executor.map(fetch_chunk, date_ranges)
            all_items = [item for chunk_items in results for item in chunk_items]

        return ItemCollection(all_items)

    def _split_datetime(
        self,
        datetime_param: DatetimeInput,
        days_per_chunk: Union[int, Literal["auto"]],
    ) -> list[str]:
        """
        Split a datetime range into smaller chunks.

        Parameters
        ----------
        datetime_param : str | tuple
            Datetime range as "start/end" string or tuple of (start, end).
        days_per_chunk : int | "auto"
            Number of days per chunk, or "auto" for automatic calculation.

        Returns
        -------
        list[str]
            List of datetime range strings in "start/end" format.
        """
        if isinstance(datetime_param, tuple):
            start_str = self._to_iso_string(datetime_param[0])
            end_str = self._to_iso_string(datetime_param[1])
        elif isinstance(datetime_param, str) and "/" in datetime_param:
            start_str, end_str = datetime_param.split("/")
            if start_str == ".." or end_str == "..":
                raise ValueError(
                    "days_per_chunk cannot be used with open-ended datetime intervals (containing '..'); "
                    "remove days_per_chunk to use standard search"
                )
        else:
            return [datetime_param if isinstance(datetime_param, str) else self._to_iso_string(datetime_param)]

        start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        total_days = (end - start).days

        if days_per_chunk == "auto":
            freq_days = max(1, total_days // 10)
        else:
            freq_days = int(days_per_chunk)
            if freq_days <= 0:
                raise ValueError(f"days_per_chunk must be a positive integer, got {days_per_chunk}")

        if total_days <= freq_days:
            return [f"{start_str}/{end_str}"]

        freq = timedelta(days=freq_days)
        one_microsecond = timedelta(microseconds=1)
        date_ranges = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + freq, end)
            date_ranges.append(f"{chunk_start.isoformat()}/{chunk_end.isoformat()}")
            chunk_start = chunk_end + one_microsecond

        date_ranges.reverse()  # executor.map preserves input order; reverse to match standard search result ordering
        return date_ranges

    def _to_iso_string(self, dt: Union[str, datetime]) -> str:
        """Convert datetime to ISO format string."""
        if isinstance(dt, str):
            return dt
        return dt.isoformat()
