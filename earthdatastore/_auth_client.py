import json
import time
from abc import ABC, abstractmethod

import requests


class Authentication(ABC):
    """
    Abstract base class for all authentication methods.

    This class serves as a template for different authentication methods. Each subclass should
    implement the `authenticate` method according to its specific protocol, updating internal state
    with the new credentials. It is not intended to return these values directly.

    Methods:
    -------
    authenticate():
        Abstract method to be implemented by subclasses. This method should handle
        the authentication process and update internal state such as the token and expiry time.

    get_token() -> str:
        Returns a valid access token. This method checks if the current token is valid
        and if not, it re-authenticates to obtain a new token.
    """

    @abstractmethod
    def authenticate(self) -> None:
        """
        An abstract method to handle authentication and update internal state with new credentials.

        This method must be implemented by subclasses to perform the authentication process specific to
        the authentication method being used. It should update internal state, such as the token and
        its expiry, rather than returning these details.
        """
        pass

    @abstractmethod
    def get_token(self) -> str:
        """
        Returns a valid access token. This method should check the token's validity and
        re-authenticate if necessary to ensure the token is still valid before returning it.

        Returns:
        -------
        str:
            The valid access token.
        """
        pass


class CognitoAuth(Authentication):
    """
    Authentication class for AWS Cognito using the OAuth 2.0 client credentials flow.

    This class exchanges client credentials (client_id and client_secret) for an access token
    at the Cognito token endpoint and handles token renewal when necessary.

    Attributes:
    ----------
    client_id: str
        The client ID for AWS Cognito.
    client_secret: str
        The client secret for AWS Cognito.
    token_url: str
        The token endpoint URL to retrieve the access token.
    token: str, optional
        The current valid access token, initially None.
    expiry: float
        The timestamp of when the current token expires.
    """

    def __init__(self, client_id: str, client_secret: str, token_url: str):
        """
        Initializes the authentication class with the necessary credentials and token endpoint.

        Parameters:
        ----------
        client_id : str
            The client ID for AWS Cognito.
        client_secret : str
            The client secret for AWS Cognito.
        token_url : str
            The token endpoint URL to retrieve the access token.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.token = None
        self.expiry = 0

    def authenticate(self) -> None:
        """
        Authenticates with AWS Cognito using client credentials and retrieves an access token.

        This method initializes a session and configures it to use HTTP Basic Authentication with the provided
        client_id and client_secret. It performs an HTTP POST request to the Cognito token endpoint, specifically
        requesting an access token via the client credentials grant type.

        Raises:
        -------
        ValueError:
            If the authentication request fails, returns an invalid response, or if the response cannot be
            parsed as JSON.
        """
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        body = {"grant_type": "client_credentials"}

        try:
            session = requests.Session()
            session.auth = (self.client_id, self.client_secret)

            response = session.post(self.token_url, headers=headers, data=body)
            response.raise_for_status()

            token_data = response.json()

            if "access_token" not in token_data:
                raise ValueError(f"Invalid response from Cognito: {token_data}")

            self.expiry = time.time() + token_data["expires_in"]
            self.token = token_data["access_token"]

        except requests.RequestException as e:
            raise ValueError(f"Authentication failed: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON response: {e}")

    def get_token(self) -> str:
        """
        Retrieves and returns a valid access token. If the current token is expired or not set,
        it re-authenticates to get a new token.

        Returns:
        -------
        str:
            A valid access token.
        """
        if not self.token or time.time() >= self.expiry:
            self.authenticate()
        if not self.token:
            raise ValueError("Failed to authenticate and retrieve token")

        return self.token
