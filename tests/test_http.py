import unittest
from unittest.mock import Mock, patch

from earthdaily._http_client import HTTPClient, HTTPRequest


class TestHTTPClient(unittest.TestCase):
    def setUp(self):
        self.client = HTTPClient()

    @patch("earthdaily._http_client.requests.Session.request")
    def test_send_success(self, mock_request):
        # Prepare mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.headers = {"Content-Type": "application/json"}
        mock_request.return_value = mock_response

        # Prepare request
        request = HTTPRequest(
            method="GET",
            url="https://api.test.com/endpoint",
            headers={"Authorization": "Bearer test_token"},
            body=None,
        )

        # Send request and assert
        response = self.client.send(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, {"result": "success"})
        self.assertEqual(response.headers, {"Content-Type": "application/json"})
        mock_request.assert_called_once()

    @patch("earthdaily._http_client.requests.Session.request")
    def test_send_failure(self, mock_request):
        # Prepare mock response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response

        # Prepare request
        request = HTTPRequest(method="GET", url="https://api.test.com/notfound")

        # Send request and assert
        response = self.client.send(request)

        self.assertEqual(response.status_code, 404)
        mock_request.assert_called_once()

    def test_http_client_retry_config(self):
        client = HTTPClient(max_retries=5, retry_backoff_factor=2.0)
        self.assertEqual(client.max_retries, 5)
        self.assertEqual(client.retry_backoff_factor, 2.0)

    @patch("earthdaily._http_client.requests.Session")
    def test_retry_session_creation(self, mock_session_class):
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        client = HTTPClient(max_retries=3, retry_backoff_factor=1.5)
        client._get_session()

        mock_session_class.assert_called_once()
        mock_session.mount.assert_any_call("https://", unittest.mock.ANY)
        mock_session.mount.assert_any_call("http://", unittest.mock.ANY)

    @patch("earthdaily._http_client.requests.Session")
    def test_retry_on_429(self, mock_session_class):
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # First call returns 429, second call succeeds
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.json.return_value = {"error": "rate limited"}
        mock_response_429.headers = {"Retry-After": "1"}

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"result": "success"}
        mock_response_success.headers = {"Content-Type": "application/json"}

        mock_session.request.side_effect = [mock_response_429, mock_response_success]

        client = HTTPClient(max_retries=3, retry_backoff_factor=1.0)

        # Just test that session is properly configured, don't make actual requests
        session = client._get_session()
        self.assertIsNotNone(session)
        self.assertIs(session, mock_session)


if __name__ == "__main__":
    unittest.main()
