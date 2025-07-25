from earthdaily._api_requester import APIRequester
from earthdaily.legacy.core import options

from . import datasets, earthdatastore  # noqa: F401
from .accessor import (  # noqa: F401
    __EarthDailyAccessorDataArray,
    __EarthDailyAccessorDataset,
)

__all__ = ["options"]


def EarthDataStore(
    api_requester: APIRequester,
    presign_urls: bool = True,
    asset_proxy_enabled: bool = False,
) -> earthdatastore.Auth:
    """
    Open earth data store connection to allow for datacube requests.
    Try to read Earth Data Store credentials from multiple sources, in the following order:
        - from input credentials stored in a given JSON file
        - from input credentials stored in a given TOML file
        - from environement variables
        - from the $HOME/.earthdaily/credentials TOML file and a given profile
        - from the $HOME/.earthdaily/credentials TOML file and the "default" profile

    Parameters
    ----------
    path : Path, optional
        The path to the TOML file containing the Earth Data Store credentials.
        Uses "$HOME/.earthdaily/credentials" by default.
    profile : profile, optional
        Name of the profile to use in the TOML file.
        Uses "default" by default.
    asset_proxy_enabled : bool, optional
        If True, the asset proxy URLs will be returned instead of presigned URLs.
        Both asset_proxy_enabled and presign_urls cannot be True at the same time. asset_proxy_enabled takes precedence.
        Uses False by default.

    Returns
    -------
    Auth
        A :class:`earthdatastore.Auth` instance
    """
    return earthdatastore.Auth(
        api_requester=api_requester,
        presign_urls=presign_urls,
        asset_proxy_enabled=asset_proxy_enabled,
    )
