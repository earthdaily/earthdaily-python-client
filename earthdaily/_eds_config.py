import os
from dataclasses import dataclass, field


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
    - `EDS_API_URL`: The URL used for interacting with the API’s endpoints.

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
    pre_sign_urls: bool, optional
        A flag indicating whether to use pre-signed URLs for asset access. Defaults to True.

    Raises:
    -------
    ValueError
        If any of the required parameters (client_id, client_secret, token_url) are not provided either via input
        or environment variables.
    """

    client_id: str = field(default_factory=lambda: os.getenv("EDS_CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: os.getenv("EDS_SECRET", ""))
    token_url: str = field(default_factory=lambda: os.getenv("EDS_AUTH_URL", ""))
    base_url: str = field(default_factory=lambda: os.getenv("EDS_API_URL", "https://api.earthdaily.com"))

    # Platform specific configurations
    pre_sign_urls: bool = True

    def __post_init__(self):
        """Validate that required fields are provided and raise errors if not."""
        missing_fields = []

        if not self.client_id:
            missing_fields.append("client_id (or EDS_CLIENT_ID)")
        if not self.client_secret:
            missing_fields.append("client_secret (or EDS_SECRET)")
        if not self.token_url:
            missing_fields.append("token_url (or EDS_AUTH_URL)")

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
