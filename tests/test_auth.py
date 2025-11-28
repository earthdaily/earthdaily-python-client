import json
import unittest
from unittest.mock import Mock, patch

import requests

from earthdaily._auth_client import Authentication


class TestAuth(unittest.TestCase):
    def setUp(self):
        self.auth = Authentication("client_id", "client_secret", "token_url")

    @patch("earthdaily._auth_client.requests.Session.post")
    def test_authenticate_success(self, mock_post):
        # Prepare mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "test_token", "expires_in": 3600}
        mock_post.return_value = mock_response

        # Call the method and assert
        token = self.auth.get_token()
        self.assertEqual(token, "test_token")
        mock_post.assert_called_once()

    @patch("earthdaily._auth_client.requests.Session.post")
    def test_authenticate_failure_http_error(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "invalid_client"}

        http_error = requests.exceptions.HTTPError("400 Client Error: Bad Request")
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_post.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            self.auth.authenticate()

        self.assertIn("Authentication failed", str(context.exception))
        self.assertIn("HTTP 400", str(context.exception))

    @patch("earthdaily._auth_client.requests.Session.post")
    def test_authenticate_401_unauthorized(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 401

        http_error = requests.exceptions.HTTPError("401 Client Error: Unauthorized")
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_post.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            self.auth.authenticate()

        exception_msg = str(context.exception)
        self.assertIn("401 Unauthorized", exception_msg)
        self.assertIn("Invalid credentials", exception_msg)
        self.assertIn("EDS_CLIENT_ID", exception_msg)

    @patch("earthdaily._auth_client.requests.Session.post")
    def test_authenticate_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with self.assertRaises(ValueError) as context:
            self.auth.authenticate()

        exception_msg = str(context.exception)
        self.assertIn("Unable to connect", exception_msg)
        self.assertIn("EDS_AUTH_URL", exception_msg)

    @patch("earthdaily._auth_client.requests.Session.post")
    def test_authenticate_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Connection timeout")

        with self.assertRaises(ValueError) as context:
            self.auth.authenticate()

        exception_msg = str(context.exception)
        self.assertIn("timeout", exception_msg.lower())
        self.assertIn("EDS_AUTH_URL", exception_msg)

    @patch("earthdaily._auth_client.requests.Session.post")
    def test_authenticate_invalid_json(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            self.auth.authenticate()

        exception_msg = str(context.exception)
        self.assertIn("Invalid JSON response", exception_msg)

    @patch("earthdaily._auth_client.requests.Session.post")
    def test_authenticate_missing_access_token(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"expires_in": 3600}
        mock_post.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            self.auth.authenticate()

        exception_msg = str(context.exception)
        self.assertIn("Invalid response from Auth provider", exception_msg)

    @patch("earthdaily._auth_client.requests.Session.post")
    def test_authenticate_general_request_exception(self, mock_post):
        mock_post.side_effect = requests.RequestException("Generic request error")

        with self.assertRaises(ValueError) as context:
            self.auth.authenticate()

        exception_msg = str(context.exception)
        self.assertIn("Authentication failed", exception_msg)


if __name__ == "__main__":
    unittest.main()
