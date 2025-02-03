from typing import Optional
from pathlib import Path
from . import earthdatastore, datasets
from .accessor import __EarthDailyAccessorDataArray, __EarthDailyAccessorDataset

# import warnings
# to hide warnings from rioxarray or nano seconds conversion
# warnings.filterwarnings("ignore")

__version__ = "0.4.0"


def EarthDataStore(
    json_path: Optional[Path] = None,
    toml_path: Optional[Path] = None,
    profile: Optional[str] = None,
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
        If True, the asset proxy URLs will be returned instead of pre-signed URLs.
        Both asset_proxy_enabled and presign_urls cannot be True at the same time. asset_proxy_enabled takes precedence.
        Uses False by default.

    Returns
    -------
    Auth
        A :class:`earthdatastore.Auth` instance
    """
    return earthdatastore.Auth.from_credentials(
        json_path=json_path,
        toml_path=toml_path,
        profile=profile,
        presign_urls=presign_urls,
        asset_proxy_enabled=asset_proxy_enabled,
    )
