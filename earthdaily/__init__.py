from typing import Optional
from pathlib import Path
from . import earthdatastore, datasets
from .accessor import EarthDailyAccessorDataArray, EarthDailyAccessorDataset

# import warnings
# to hide warnings from rioxarray or nano seconds conversion
# warnings.filterwarnings("ignore")

__version__ = "0.2.2"

def EarthDataStore(
    json_path: Optional[Path] = None,
    toml_path: Optional[Path] = None,
    profile: Optional[str] = None,
    presign_urls: bool = True,
) -> earthdatastore.Auth:
    """
    TODO
    """
    return earthdatastore.Auth.from_credentials(
        json_path = json_path,
        toml_path = toml_path,
        profile = profile,
        presign_urls = presign_urls
    )
