import json
import unittest
from unittest.mock import Mock, patch

from earthdaily.exceptions import EDSAPIError
from earthdaily.platform._stac_item import StacItemService


class TestStacItemService(unittest.TestCase):
    def setUp(self):
        self.mock_api_requester = Mock()
        self.mock_api_requester.base_url = "https://api.example.com"
        self.stac_item_service = StacItemService(self.mock_api_requester)

    def test_create_item_success_dict_format(self):
        """Test successful item creation with DICT return format."""
        mock_response_data = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "test_item_id",
            "properties": {"datetime": "2023-01-01T00:00:00Z"},
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "links": [],
            "assets": {},
        }

        item_data = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "test_item_id",
            "properties": {"datetime": "2023-01-01T00:00:00Z"},
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "links": [],
            "assets": {},
        }

        self.mock_api_requester.send_request.return_value.status_code = 201
        self.mock_api_requester.send_request.return_value.body = mock_response_data

        result = self.stac_item_service.create_item(
            collection_id="test_collection", item_data=item_data, return_format="dict"
        )

        self.assertEqual(result, mock_response_data)

        # Verify the request was made correctly
        expected_url = "https://api.example.com/platform/v1/stac/collections/test_collection/items"
        self.mock_api_requester.send_request.assert_called_once()
        actual_request = self.mock_api_requester.send_request.call_args[0][0]
        self.assertEqual(actual_request.method, "POST")
        self.assertEqual(actual_request.url, expected_url)
        self.assertEqual(actual_request.body, item_data)

    def test_create_item_success_json_format(self):
        """Test successful item creation with JSON return format."""
        mock_response_data = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "test_item_id",
            "properties": {"datetime": "2023-01-01T00:00:00Z"},
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "links": [],
            "assets": {},
        }

        item_data = {"id": "test_item", "type": "Feature"}

        self.mock_api_requester.send_request.return_value.status_code = 201
        self.mock_api_requester.send_request.return_value.body = mock_response_data

        result = self.stac_item_service.create_item(
            collection_id="test_collection", item_data=item_data, return_format="json"
        )

        expected_json = json.dumps(mock_response_data, indent=2)
        self.assertEqual(result, expected_json)

    @patch("earthdaily.platform._stac_item.pystac")
    def test_create_item_success_pystac_format(self, mock_pystac):
        """Test successful item creation with PYSTAC return format."""
        mock_response_data = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "test_item_id",
            "properties": {"datetime": "2023-01-01T00:00:00Z"},
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "links": [],
            "assets": {},
        }

        item_data = {"id": "test_item", "type": "Feature"}
        mock_pystac_item = Mock()
        mock_pystac.Item.from_dict.return_value = mock_pystac_item

        self.mock_api_requester.send_request.return_value.status_code = 201
        self.mock_api_requester.send_request.return_value.body = mock_response_data

        result = self.stac_item_service.create_item(
            collection_id="test_collection", item_data=item_data, return_format="pystac"
        )

        self.assertEqual(result, mock_pystac_item)
        mock_pystac.Item.from_dict.assert_called_once_with(mock_response_data)

    def test_create_item_api_error_400(self):
        """Test item creation with 400 API error."""
        item_data = {"id": "test_item", "type": "Feature"}

        self.mock_api_requester.send_request.return_value.status_code = 400
        self.mock_api_requester.send_request.return_value.body = {"error": "Invalid item data"}

        with self.assertRaises(EDSAPIError) as context:
            self.stac_item_service.create_item(collection_id="test_collection", item_data=item_data)

        self.assertEqual(str(context.exception), "API request failed with status 400: {'error': 'Invalid item data'}")

    def test_create_item_api_error_500(self):
        """Test item creation with 500 API error."""
        item_data = {"id": "test_item", "type": "Feature"}

        self.mock_api_requester.send_request.return_value.status_code = 500
        self.mock_api_requester.send_request.return_value.body = {"error": "Internal server error"}

        with self.assertRaises(EDSAPIError) as context:
            self.stac_item_service.create_item(collection_id="test_collection", item_data=item_data)

        self.assertIn("API request failed with status 500", str(context.exception))

    def test_create_item_default_return_format(self):
        """Test that default return format is DICT."""
        mock_response_data = {"type": "Feature", "id": "test_item"}
        item_data = {"id": "test_item", "type": "Feature"}

        self.mock_api_requester.send_request.return_value.status_code = 201
        self.mock_api_requester.send_request.return_value.body = mock_response_data

        result = self.stac_item_service.create_item(collection_id="test_collection", item_data=item_data)

        # Should return the dict directly since default is "dict"
        self.assertEqual(result, mock_response_data)

    def test_create_item_unsupported_return_format(self):
        """Test item creation with unsupported return format."""
        item_data = {"id": "test_item", "type": "Feature"}

        with self.assertRaises(ValueError) as context:
            self.stac_item_service.create_item(
                collection_id="test_collection", item_data=item_data, return_format="invalid_format"
            )

        self.assertIn("Unsupported return format: invalid_format", str(context.exception))

    def test_create_item_empty_collection_id(self):
        """Test item creation with empty collection ID."""
        item_data = {"id": "test_item", "type": "Feature"}
        mock_response_data = {"type": "Feature", "id": "test_item"}

        self.mock_api_requester.send_request.return_value.status_code = 201
        self.mock_api_requester.send_request.return_value.body = mock_response_data

        self.stac_item_service.create_item(collection_id="", item_data=item_data)

        # Verify the URL includes empty collection_id
        expected_url = "https://api.example.com/platform/v1/stac/collections//items"
        actual_request = self.mock_api_requester.send_request.call_args[0][0]
        self.assertEqual(actual_request.url, expected_url)

    def test_create_item_complex_item_data(self):
        """Test item creation with complex item data."""
        complex_item_data = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "complex_item",
            "properties": {
                "datetime": "2023-01-01T00:00:00Z",
                "platform": "test-platform",
                "instruments": ["sensor1", "sensor2"],
                "gsd": 10.0,
            },
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            "links": [{"rel": "self", "href": "https://example.com/item/complex_item"}],
            "assets": {"thumbnail": {"href": "https://example.com/thumbnail.jpg", "type": "image/jpeg"}},
        }

        self.mock_api_requester.send_request.return_value.status_code = 201
        self.mock_api_requester.send_request.return_value.body = complex_item_data

        result = self.stac_item_service.create_item(collection_id="complex_collection", item_data=complex_item_data)

        self.assertEqual(result, complex_item_data)

        # Verify the complex data was sent correctly
        actual_request = self.mock_api_requester.send_request.call_args[0][0]
        self.assertEqual(actual_request.body, complex_item_data)


if __name__ == "__main__":
    unittest.main()
