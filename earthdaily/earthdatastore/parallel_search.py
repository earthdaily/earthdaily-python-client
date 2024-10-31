from functools import wraps
from typing import Callable, TypeVar, Any, Optional, Union, Tuple, List

from datetime import datetime
from itertools import chain
from joblib import Parallel, delayed
import logging
import time
import numpy as np
from pystac.item_collection import ItemCollection
import pandas as pd
from pandas import Timestamp, Timedelta, DatetimeIndex
from pystac_client.item_search import ItemSearch

T = TypeVar("T")

DatetimeRange = Tuple[Union[datetime, Timestamp], Union[datetime, Timestamp]]
DateRangeList = List[DatetimeRange]


class NoItemsFoundError(Exception):
    """Exception raised when no items are found during search operation.

    This exception is raised when a parallel search operation yields no results,
    indicating that the search criteria did not match any items in the dataset.
    """

    pass


def datetime_to_str(dt_range: DatetimeRange) -> Tuple[str, str]:
    """Convert a datetime range to a tuple of formatted strings.

    Parameters
    ----------
    dt_range : tuple of (datetime or Timestamp)
        A tuple containing start and end datetimes to be converted.

    Returns
    -------
    tuple of str
        A tuple containing two strings representing the formatted start and end dates.

    Notes
    -----
    This function relies on ItemSearch._format_datetime internally to perform the
    actual formatting. The returned strings are split from a forward-slash separated
    string format.

    Examples
    --------
    >>> start = pd.Timestamp('2023-01-01')
    >>> end = pd.Timestamp('2023-12-31')
    >>> datetime_to_str((start, end))
    ('2023-01-01', '2023-12-31')
    """
    formatted = ItemSearch(url=None)._format_datetime(dt_range)
    start, end = formatted.split("/")
    return start, end


def datetime_split(
    dt_range: DatetimeRange, freq: Union[str, int, Timedelta] = "auto", n_jobs: int = 10
) -> Union[DatetimeRange, Tuple[DateRangeList, Timedelta]]:
    """Split a datetime range into smaller chunks based on specified frequency.

    Parameters
    ----------
    dt_range : tuple of (datetime or Timestamp)
        A tuple containing the start and end datetimes to split.
    freq : str or int or Timedelta, default="auto"
        The frequency to use for splitting the datetime range.
        If "auto", frequency is calculated based on the total date range:
        It increases by 5 days for every 6 months in the range.
        If int, interpreted as number of days.
        If Timedelta, used directly as the splitting frequency.
    n_jobs : int, default=10
        Number of jobs for parallel processing (currently unused in the function
        but maintained for API compatibility).

    Returns
    -------
    Union[DatetimeRange, tuple[list[DatetimeRange], Timedelta]]
        If the date range is smaller than the frequency:
            Returns the original datetime range tuple.
        Otherwise:
            Returns a tuple containing:
            - List of datetime range tuples split by the frequency
            - The Timedelta frequency used for splitting

    Notes
    -----
    The automatic frequency calculation uses the formula:
    freq = total_days // (5 + 5 * (total_days // 183))

    This ensures that the frequency increases by 5 days for every 6-month period
    in the total date range.

    Examples
    --------
    >>> start = pd.Timestamp('2023-01-01')
    >>> end = pd.Timestamp('2023-12-31')
    >>> splits, freq = datetime_split((start, end))
    >>> len(splits)  # Number of chunks
    12

    >>> # Using fixed frequency
    >>> splits, freq = datetime_split((start, end), freq=30)  # 30 days
    >>> freq
    Timedelta('30 days')
    """
    # Convert input dates to pandas Timestamps
    start, end = [pd.Timestamp(date) for date in datetime_to_str(dt_range)]
    date_diff = end - start

    # Calculate or convert frequency
    if freq == "auto":
        # Calculate automatic frequency based on total range
        total_days = date_diff.days
        months_factor = total_days // 183  # 183 days ≈ 6 months
        freq = Timedelta(days=(total_days // (5 + 5 * months_factor)))
    elif isinstance(freq, (int, str)):
        freq = Timedelta(days=int(freq))
    elif not isinstance(freq, Timedelta):
        raise TypeError("freq must be 'auto', int, or Timedelta")

    # Return original range if smaller than frequency
    if date_diff.days < freq.days or freq.days <= 1:
        return dt_range, freq

    # Generate date ranges
    date_ranges = [
        (chunk, min(chunk + freq, end))
        for chunk in pd.date_range(start, end, freq=freq)[:-1]
    ]

    return date_ranges, freq


def parallel_search(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for parallelizing search operations across datetime ranges.

    This decorator enables parallel processing of search operations by splitting the
    datetime range into batches. It automatically handles parallel execution when
    conditions are met (multiple batches or large date range) and falls back to
    sequential processing otherwise.

    Parameters
    ----------
    func : callable
        The search function to be parallelized. Should accept the following kwargs:
        - datetime : tuple of datetime
            Range of dates to search
        - batch_days : int or "auto", optional
            Number of days per batch for splitting
        - n_jobs : int, optional
            Number of parallel jobs. Use -1 or >10 for maximum of 10 jobs
        - raise_no_items : bool, optional
            Whether to raise exception when no items found

    Returns
    -------
    callable
        Wrapped function that handles parallel execution of the search operation.

    Notes
    -----
    The wrapped function preserves the same interface as the original function
    but adds parallel processing capabilities based on the following parameters
    in kwargs:
    - batch_days : Controls the size of datetime batches
    - n_jobs : Controls the number of parallel jobs (max 10)
    - datetime : Required for parallel execution

    The parallel execution uses threading backend from joblib.

    See Also
    --------
    joblib.Parallel : Used for parallel execution
    datetime_split : Helper function for splitting datetime ranges

    Examples
    --------
    >>> @parallel_search
    ... def search_items(query, datetime=None, batch_days="auto", n_jobs=1):
    ...     # Search implementation
    ...     return items
    >>>
    >>> # Will execute in parallel if conditions are met
    >>> items = search_items("query",
    ...                     datetime=(start_date, end_date),
    ...                     batch_days=30,
    ...                     n_jobs=4)
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()

        # Set default parameters
        batch_days = kwargs.setdefault("batch_days", "auto")
        n_jobs = kwargs.setdefault("n_jobs", -1)
        dt_range = kwargs.get("datetime")

        should_parallelize = _should_run_parallel(dt_range, batch_days, n_jobs)

        if should_parallelize:
            items = _run_parallel_search(func, args, kwargs, dt_range, batch_days)
        else:
            items = func(*args, **kwargs)

        execution_time = np.round(time.time() - start_time, 3)
        logging.info(f"Search/load items: {execution_time}s")

        return items

    return wrapper


def _should_run_parallel(
    dt_range: Optional[tuple[datetime, datetime]], batch_days: Any, n_jobs: int
) -> bool:
    """Check if parallel execution should be used based on input parameters.
    Parameters
    ----------
    dt_range : tuple of datetime or None
        The start and end datetime for the search range
    batch_days : int or "auto" or None
        Number of days per batch for splitting the datetime range
    n_jobs : int
        Number of parallel jobs requested
    Returns
    -------
    bool
        True if parallel execution should be used, False otherwise
    Notes
    -----
    Parallel execution is used when all of the following conditions are met:
    - dt_range is provided and not None
    - batch_days is not None
    - n_jobs > 1
    - Either multiple date ranges exist or the total days exceed batch_days
    """
    # Check for basic conditions that prevent parallel execution
    if not dt_range or batch_days is None or n_jobs <= 1:
        return False

    # Split the datetime range
    date_ranges, freq = datetime_split(dt_range, batch_days)

    # Check if splitting provides meaningful parallelization
    delta_days = (date_ranges[-1][-1] - date_ranges[0][0]).days
    return len(date_ranges) > 1 or delta_days > batch_days


def _run_parallel_search(
    func: Callable,
    args: tuple,
    kwargs: dict,
    dt_range: tuple[datetime, datetime],
    batch_days: Any,
) -> T:
    """Execute the search function in parallel across datetime batches.

    Parameters
    ----------
    func : callable
        The search function to be executed in parallel
    args : tuple
        Positional arguments to pass to the search function
    kwargs : dict
        Keyword arguments to pass to the search function
    dt_range : tuple of datetime
        The start and end datetime for the search range
    batch_days : int or "auto"
        Number of days per batch for splitting the datetime range

    Returns
    -------
    T
        Combined results from all parallel executions

    Raises
    ------
    NoItemsFoundError
        If no items are found across all parallel executions

    Notes
    -----
    This function:
    1. Splits the datetime range into batches
    2. Configures parallel execution parameters
    3. Runs the search function in parallel using joblib
    4. Combines results from all parallel executions

    The maximum number of parallel jobs is capped at 10, and -1 is converted to 10.
    """
    date_ranges, freq = datetime_split(dt_range, batch_days)

    logging.info(
        f"Search parallel with {kwargs['n_jobs']} jobs, split every {freq.days} days."
    )

    # Prepare kwargs for parallel execution
    parallel_kwargs = kwargs.copy()
    parallel_kwargs.pop("datetime")
    parallel_kwargs["raise_no_items"] = False

    # Handle n_jobs special case: -1 should become 10
    n_jobs = parallel_kwargs.get("n_jobs", 10)
    parallel_kwargs["n_jobs"] = 10 if (n_jobs == -1 or n_jobs > 10) else n_jobs

    # Execute parallel search
    results = Parallel(n_jobs=parallel_kwargs["n_jobs"], backend="threading")(
        delayed(func)(*args, datetime=dt, **parallel_kwargs) for dt in date_ranges
    )

    # Combine results
    items = ItemCollection(chain(*results))
    if not items:
        raise NoItemsFoundError("No items found in parallel search")

    return items
