from earthdaily._api_requester import APIRequester
from earthdaily.legacy.core import options

from . import datasets, earthdatastore  # noqa: F401
from .accessor import (  # noqa: F401
    __EarthDailyAccessorDataArray,
    __EarthDailyAccessorDataset,
)

__all__ = ["options"]


def EarthDataStore(api_requester: APIRequester) -> earthdatastore.Auth:
    """
    Create an Auth client for interacting with legacy EarthDataStore APIs.

    Parameters
    ----------
    api_requester: APIRequester
        An instance of APIRequester used to send HTTP requests to the EDS API.

    Returns
    -------
    Auth
        A :class:`earthdatastore.Auth` instance
    """
    return earthdatastore.Auth(
        api_requester=api_requester,
    )
