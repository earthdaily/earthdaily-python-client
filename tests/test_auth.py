import unittest
from unittest.mock import Mock, patch

from earthdaily._auth_client import CognitoAuth


class TestCognitoAuth(unittest.TestCase):
    def setUp(self):
        self.auth = CognitoAuth("client_id", "client_secret", "token_url")

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
    def test_authenticate_failure(self, mock_post):
        # Prepare mock response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "invalid_client"}
        mock_post.return_value = mock_response

        # Call the method and assert
        with self.assertRaises(ValueError) as context:
            self.auth.authenticate()

        self.assertIn("Invalid response from Cognito", str(context.exception))


if __name__ == "__main__":
    unittest.main()
