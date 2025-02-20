import unittest
from unittest.mock import Mock, patch

from earthdaily._http_client import HTTPClient, HTTPRequest


class TestHTTPClient(unittest.TestCase):
    def setUp(self):
        self.client = HTTPClient()

    @patch("earthdaily._http_client.requests.request")
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

    @patch("earthdaily._http_client.requests.request")
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


if __name__ == "__main__":
    unittest.main()
