from pathlib import Path

from earthdaily._api_requester import APIRequester
from earthdaily._auth_client import Authentication
from earthdaily._eds_config import EDSConfig
from earthdaily._platform_cookie import DEFAULT_TTL_SECONDS, PlatformCookieManager

try:
    from earthdaily.platform import PlatformService

    _HAS_PLATFORM = True
except ImportError:
    _HAS_PLATFORM = False

try:
    from earthdaily.internal import InternalService  # type: ignore[import]

    _HAS_INTERNAL = True
except ImportError:
    _HAS_INTERNAL = False

try:
    from earthdaily.ordering import OrderingService

    _HAS_ORDERING = True
except ImportError:
    _HAS_ORDERING = False

try:
    from earthdaily.datacube import DatacubeService

    _HAS_DATACUBE = True
except ImportError:
    _HAS_DATACUBE = False


class EDSClient:
    """
    EDSClient is the main entry point for interacting with the Earth Data Store (EDS).

    This client manages authentication and facilitates service interactions through the APIRequester,
    using configurations specified in EDSConfig.

    The client uses the client_credentials flow to access the EDS services.

    Attributes:
    ----------
    config: EDSConfig
        The configuration object containing settings such as client_id, client_secret, token_url, and base_url.
    auth: Authentication
        The authentication object, which handles obtaining and managing authentication tokens.
    api_requester: APIRequester
        An instance that handles sending authenticated requests to the EDS API.
    """

    def __init__(self, config: EDSConfig):
        """
        Initialize the EDSClient with the provided configuration.

        This constructor sets up the authentication and API request handling components based on the
        provided EDSConfig settings. Authentication is performed immediately to validate credentials.

        Parameters:
        ----------
        config: EDSConfig
            The configuration object containing client_id, client_secret, token_url, and base_url.

        Raises:
        -------
        ValueError:
            If authentication fails due to invalid credentials, connection errors, or other issues.
        """
        self.config = config
        self.auth = self._create_auth() if not config.bypass_auth else None
        self.api_requester = APIRequester(config=config, auth=self.auth)
        self._platform_cookie_manager: PlatformCookieManager | None = None

        if not config.bypass_auth and self.auth:
            # Validate credentials on initialization to fail fast
            self.auth.authenticate()

    def _create_auth(self) -> Authentication:
        """
        Creates the authentication mechanism.

        Returns:
        -------
        Authentication:
            An instance of an authentication class.
        """
        return Authentication(self.config.client_id, self.config.client_secret, self.config.token_url)

    @property
    def platform(self):
        """
        Lazily initializes and returns the PlatformService for interacting with platform-related API endpoints.

        Returns:
        -------
        PlatformService:
            The service that interacts with platform-specific API operations.

        Raises:
        -------
        ImportError:
            If the platform module dependencies are not available.
        """
        if not _HAS_PLATFORM:
            raise ImportError(
                "Platform functionality requires additional dependencies. "
                "Install with: pip install 'earthdaily[platform]'"
            )
        if not hasattr(self, "_platform_service"):
            self._platform_service = PlatformService(self.api_requester, self.config.asset_access_mode)
        return self._platform_service

    @property
    def internal(self):
        """
        Lazily initializes and returns the InternalService for interacting with internal API endpoints.

        Returns:
        -------
        InternalService:
            The service that interacts with internal API operations.
        Raises:
        -------
        NotImplementedError:
            If the internal module is not available in this build of EDSClient.
        """
        if not _HAS_INTERNAL:
            raise NotImplementedError("The 'internal' module is not available in this build of EDSClient.")
        if not hasattr(self, "_internal_service"):
            self._internal_service = InternalService(self.api_requester)
        return self._internal_service

    @property
    def ordering(self):
        """
        Lazily initializes and returns the OrderingService for interacting with ordering API endpoints.

        Returns:
        -------
        OrderingService:
            The service that interacts with ordering operations.

        Raises:
        -------
        ImportError:
            If the ordering module dependencies are not available.
        """
        if not _HAS_ORDERING:
            raise ImportError(
                "Ordering functionality requires additional dependencies. "
                "Install with: pip install 'earthdaily[ordering]'"
            )
        if not hasattr(self, "_ordering_service"):
            self._ordering_service = OrderingService(self.api_requester)
        return self._ordering_service

    @property
    def datacube(self):
        """
        Lazily initializes and returns the DatacubeService for creating and managing datacubes.

        Returns:
        -------
        DatacubeService:
            The service that handles datacube creation and operations.

        Raises:
        -------
        ImportError:
            If the datacube module dependencies are not available.
        """
        if not _HAS_DATACUBE:
            raise ImportError(
                "Datacube functionality requires additional dependencies. "
                "Install with: pip install 'earthdaily[datacube]'"
            )
        if not hasattr(self, "_datacube_service"):
            self._datacube_service = DatacubeService()
        return self._datacube_service

    def get_platform_cookie_file(
        self,
        path: str | Path,
        force_refresh: bool = False,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> Path:
        """
        Fetch a platform asset cookie and write it to a Netscape-format file for GDAL.

        The returned file can be passed to GDAL/rasterio via the
        `GDAL_HTTP_COOKIEFILE` environment variable (for example, through
        `rasterio.Env(GDAL_HTTP_COOKIEFILE=...)` or `odc.stac.configure_rio(...)`
        with dask) so that authenticated proxy asset URLs can be opened
        without an `Authorization` header.

        This method requires `EDSConfig(asset_access_mode="proxy-urls",
        platform_cookie_enabled=True)`. If `platform_cookie_auto_refresh=True`,
        a background daemon thread is started on first call to refresh the
        cookie file before it expires.

        Parameters
        ----------
        path : str | Path
            Required path to the cookie file. The parent directory must already
            exist; this method does not create directories. Choose a location
            outside your repository (for example `/tmp/earthdaily-platform-cookies.txt`)
            to avoid accidentally committing the credential-like file.
        force_refresh : bool, default False
            If True, force a fresh cookie fetch even if the current one is
            still valid.
        ttl_seconds : int, default 3600
            Desired cookie lifetime in seconds. Sent to the cookie endpoint as
            the `max_age` field. The server may clamp this value.

        Returns
        -------
        Path
            The resolved cookie file path.

        Raises
        ------
        ValueError
            If cookie support is not enabled, the client is not in proxy-urls
            mode, `path` is missing, or the parent directory does not exist.
        EDSAPIError
            If the cookie endpoint returns an error or the response is missing
            the expected `PLATFORM_ASSET` cookie.
        """
        manager = self._get_or_create_platform_cookie_manager()
        return manager.get_cookie_file(path=path, force_refresh=force_refresh, ttl_seconds=ttl_seconds)

    def get_platform_cookie(self) -> str:
        """
        Return the current platform asset cookie value, refreshing if needed.

        Useful for non-GDAL callers that want to set a `Cookie` header directly.
        Requires the same configuration as `get_platform_cookie_file()`.

        Returns
        -------
        str
            The raw cookie token value.

        Raises
        ------
        ValueError
            If cookie support is not enabled or the client is not in
            proxy-urls mode.
        EDSAPIError
            If the cookie endpoint returns an error.
        """
        manager = self._get_or_create_platform_cookie_manager()
        return manager.get_cookie_value()

    def start_platform_cookie_refresh(self) -> None:
        """
        Start the background daemon thread that refreshes the platform cookie file.

        Has no effect if already running. Requires a prior successful call to
        `get_platform_cookie_file(path=...)` so a target path is known.
        """
        manager = self._get_or_create_platform_cookie_manager()
        manager.start_auto_refresh()

    def stop_platform_cookie_refresh(self) -> None:
        """
        Stop the background platform cookie refresh thread if it is running.
        """
        if self._platform_cookie_manager is not None:
            self._platform_cookie_manager.stop_auto_refresh()

    def close(self) -> None:
        """
        Release client-managed resources.

        Stops the background platform cookie refresh thread and deletes the
        cookie file if one was written by this client.
        """
        if self._platform_cookie_manager is not None:
            self._platform_cookie_manager.close()

    def __enter__(self) -> "EDSClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_or_create_platform_cookie_manager(self) -> PlatformCookieManager:
        if self._platform_cookie_manager is None:
            self._platform_cookie_manager = PlatformCookieManager(api_requester=self.api_requester, config=self.config)
        return self._platform_cookie_manager
