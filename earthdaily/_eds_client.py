from earthdaily._api_requester import APIRequester
from earthdaily._auth_client import Authentication
from earthdaily._eds_config import EDSConfig

try:
    from earthdaily.legacy import EarthDataStore

    _HAS_LEGACY = True
except ImportError:
    _HAS_LEGACY = False

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
    def legacy(self):
        """
        Lazily initializes and returns the LegacyService for interacting with v0.5.x earthdaily-python-client methods.

        Returns:
        -------
        LegacyService:
            The service that interacts with legacy API operations.

        Raises:
        -------
        ImportError:
            If the legacy module dependencies are not available.
        """
        if not _HAS_LEGACY:
            raise ImportError(
                "Legacy functionality requires additional dependencies. Install with: pip install 'earthdaily[legacy]'"
            )
        if not hasattr(self, "_legacy_service"):
            self._legacy_service = EarthDataStore(self.api_requester)
        return self._legacy_service

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
