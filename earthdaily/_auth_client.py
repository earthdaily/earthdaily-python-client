import json
import time

import requests


class Authentication:
    """
    Authentication class using the OAuth 2.0 client credentials flow.

    This class exchanges client credentials (client_id and client_secret) for an access token
    at the token endpoint and handles token renewal when necessary.

    Attributes:
    ----------
    client_id: str
        client ID
    client_secret: str
        client secret
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
            client ID
        client_secret : str
            client secret
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
        Authenticates using client credentials and retrieves an access token.

        This method initializes a session and configures it to use HTTP Basic Authentication with the provided
        client_id and client_secret. It performs an HTTP POST request to the token endpoint, specifically
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
                raise ValueError(f"Invalid response from Auth provider: {token_data}")

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
