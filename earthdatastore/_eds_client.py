from earthdatastore._api_requester import APIRequester
from earthdatastore._auth_client import Authentication, CognitoAuth
from earthdatastore._eds_config import EDSConfig
from earthdatastore.agriculture import EarthDataStore
from earthdatastore.platform import PlatformService


class EDSClient:
    """
    EDSClient is the main entry point for interacting with the Earth Data Store (EDS).

    This client manages authentication and facilitates service interactions through the APIRequester,
    using configurations specified in EDSConfig.

    The client uses to the authentication strategy provided
    in the configuration, such as Cognito, to handle access to the EDS services.

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
        provided EDSConfig settings.

        Parameters:
        ----------
        config: EDSConfig
            The configuration object containing client_id, client_secret, token_url, and base_url.
        """
        self.config = config
        self.auth = self._create_auth()
        self.api_requester = APIRequester(base_url=self.config.base_url, auth=self.auth)

    def _create_auth(self) -> Authentication:
        """
        Creates the authentication mechanism based on the provided auth_method.

        Returns:
        -------
        Authentication:
            An instance of an authentication class (CognitoAuth or Auth0Auth).

        Raises:
        -------
        ValueError:
            If the provided auth_method is unsupported.
        """
        if self.config.auth_method == "cognito":
            return CognitoAuth(self.config.client_id, self.config.client_secret, self.config.token_url)
        else:
            raise ValueError(f"Unsupported auth method: {self.config.auth_method}")

    @property
    def platform(self):
        """
        Lazily initializes and returns the PlatformService for interacting with platform-related API endpoints.

        Returns:
        -------
        PlatformService:
            The service that interacts with platform-specific API operations.
        """
        if not hasattr(self, "_platform_service"):
            self._platform_service = PlatformService(self.api_requester, self.config.pre_sign_urls)
        return self._platform_service

    @property
    def agriculture(self):
        """
        TBD

        Returns:
        -------
        AgricultureService:
            TBD
        """
        if not hasattr(self, "_agriculture_service"):
            self._agriculture_service = EarthDataStore()
        return self._agriculture_service
