import json
import os
from configparser import ConfigParser
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import toml


class AssetAccessMode(str, Enum):
    """
    Enum-like class to define asset access modes.
    """

    RAW = "raw"
    PRESIGNED_URLS = "presigned-urls"
    PROXY_URLS = "proxy-urls"


@dataclass
class EDSConfig:
    """
    Configuration class for the Earth Data Store (EDS) Client.

    This class is responsible for holding configuration values needed to interact with the
    Earth Data Store API, including authentication credentials, API endpoints, and the
    authentication method. The configuration can be provided via input parameters or fetched
    from environment variables.

    The priority for each field is as follows:
    1. If a parameter is passed during initialization, it is used.
    2. If no parameter is provided, it falls back to an environment variable.
    3. If neither is available (for required fields), it raises a ValueError.

    Required Environment Variables (if not passed during initialization):
    - `EDS_CLIENT_ID`: The client ID for authentication.
    - `EDS_SECRET`: The client secret for authentication.
    - `EDS_AUTH_URL`: The token URL used for retrieving the authentication token.
    - `EDS_API_URL`: The URL used for interacting with the API's endpoints.

    Parameters:
    ----------
    client_id: str
        The client ID for authentication. If not provided, it defaults to `EDS_CLIENT_ID` from environment variables.
    client_secret: str
        The client secret for authentication.
        If not provided, it defaults to `EDS_SECRET` from environment variables.
    token_url: str
        The token URL to retrieve authentication tokens.
        If not provided, it defaults to `EDS_AUTH_URL` from environment variables.
    base_url: str, optional
        The base URL for the Earth Data Store API. Defaults to "https://api.earthdaily.com".
    json_path: str, optional
        The path to a JSON file containing configuration settings.
    toml_path: str, optional
        The path to a TOML file containing configuration settings.
    ini_path: str, optional
        The path to an INI file containing configuration settings.
    bypass_auth: bool, optional
        A flag indicating whether to bypass authentication. Defaults to False.
    asset_access_mode: AssetAccessMode, optional
        The mode of access for assets. Defaults to AssetAccessMode.PRESIGNED_URLS.

    Raises:
    -------
    ValueError
        If any of the required parameters (client_id, client_secret, token_url) are not provided either via input
        or environment variables.
    """

    client_id: str = ""
    client_secret: str = ""
    token_url: str = ""
    base_url: str = ""
    json_path: str = field(default_factory=lambda: os.getenv("EDS_JSON_PATH", ""))
    toml_path: str = field(default_factory=lambda: os.getenv("EDS_TOML_PATH", ""))
    ini_path: str = field(default_factory=lambda: os.getenv("EDS_INI_PATH", ""))
    ini_profile: str = field(default_factory=lambda: os.getenv("EDS_INI_PROFILE", "default"))
    bypass_auth: bool = False

    # Platform specific configurations
    asset_access_mode: AssetAccessMode = AssetAccessMode.PRESIGNED_URLS

    def __post_init__(self):
        """Validate that required fields are provided and raise errors if not."""
        if self.json_path:
            config = self.load_config_from_file(self.json_path)
        elif self.toml_path:
            config = self.load_config_from_file(self.toml_path)
        elif self.ini_path:
            config = self.load_config_from_file(self.ini_path)
        else:
            config = dict(os.environ)
        self.client_id = self.client_id or config.get("EDS_CLIENT_ID")
        self.client_secret = self.client_secret or config.get("EDS_SECRET")
        self.token_url = self.token_url or config.get("EDS_AUTH_URL")
        self.base_url = self.base_url or config.get("EDS_API_URL", "https://api.earthdaily.com")
        missing_fields = []

        if not self.bypass_auth:
            if not self.client_id:
                missing_fields.append("client_id (or EDS_CLIENT_ID)")
            if not self.client_secret:
                missing_fields.append("client_secret (or EDS_SECRET)")
            if not self.token_url:
                missing_fields.append("token_url (or EDS_AUTH_URL)")

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    def load_config_from_file(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file '{file_path}' does not exist.")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        with open(file_path, "r") as f:
            if ext == ".json":
                return json.load(f)
            elif ext == ".toml":
                return toml.load(f)
            elif ext == ".ini":
                return self.read_credentials_from_ini(file_path, self.ini_profile)
            else:
                raise ValueError(f"Unsupported config file type: '{ext}'. Use .json or .toml")

    @staticmethod
    def read_credentials_from_ini(ini_path: str, profile: str = "default") -> dict[str, str]:
        """
        Read credentials from an INI file.

        Parameters
        ----------
        ini_path : str
            The path to the INI file containing credentials.
        profile : str, optional
            The profile section name to read from. Defaults to 'default'.

        Returns
        -------
        dict
            Dictionary containing credentials with upper-cased keys.

        Raises
        ------
        FileNotFoundError
            If the INI file does not exist.
        ValueError
            If the specified profile is not found in the INI file.
        """
        ini_file = Path(ini_path).expanduser().resolve()

        if not ini_file.exists():
            raise FileNotFoundError(f"INI config file not found at: {ini_file}")

        config_parser = ConfigParser()
        config_parser.read(ini_file)

        if profile not in config_parser:
            available = ", ".join(config_parser.sections()) or "no profiles"
            raise ValueError(f"Profile '{profile}' not found in INI file. Available profiles: {available}")

        credentials = {key.upper(): value for key, value in config_parser[profile].items()}

        return credentials
