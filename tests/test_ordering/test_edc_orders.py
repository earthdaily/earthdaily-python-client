from __future__ import annotations

import unittest
from unittest.mock import Mock

import pystac

from earthdaily._api_requester import APIRequester
from earthdaily._http_client import HTTPResponse
from earthdaily.exceptions import EDSAPIError
from earthdaily.ordering._edc_orders import (
    EdcLineItem,
    EdcLineItemResponse,
    EdcOrder,
    EdcOrderLineItem,
    EdcOrderRequest,
    EdcOrdersService,
    EdcProcessedProduct,
    EdcProductType,
)


class TestEdcOrderLineItem(unittest.TestCase):
    def test_to_dict(self):
        line_item = EdcOrderLineItem(input_uuid="test_uuid", product_type="VISUAL_RGB", order_name="test_order")
        expected = {"input_uuid": "test_uuid", "product_type": "VISUAL_RGB", "order_name": "test_order"}
        self.assertEqual(line_item.to_dict(), expected)


class TestEdcOrderRequest(unittest.TestCase):
    def test_to_dict(self):
        line_items = [
            EdcOrderLineItem(input_uuid="uuid1", product_type="VISUAL_RGB", order_name="order1"),
            EdcOrderLineItem(input_uuid="uuid2", product_type="VISUAL_RGB", order_name="order2"),
        ]
        request = EdcOrderRequest(line_items=line_items)
        result = request.to_dict()

        self.assertIn("line_items", result)
        self.assertEqual(len(result["line_items"]), 2)
        self.assertEqual(result["line_items"][0]["input_uuid"], "uuid1")
        self.assertEqual(result["line_items"][1]["input_uuid"], "uuid2")


class TestEdcProcessedProduct(unittest.TestCase):
    def test_from_dict(self):
        data = {
            "processed_product_uuid": "prod_uuid",
            "state": "COMPLETED",
            "processed_date": "2024-01-01T00:00:00Z",
            "sqkm": 100.5,
            "output_type": "VISUAL_RGB",
            "collection": "test_collection",
            "stac_datetime": "2024-01-01T00:00:00Z",
        }
        product = EdcProcessedProduct.from_dict(data)

        self.assertEqual(product.processed_product_uuid, "prod_uuid")
        self.assertEqual(product.state, "COMPLETED")
        self.assertEqual(product.processed_date, "2024-01-01T00:00:00Z")
        self.assertEqual(product.sqkm, 100.5)
        self.assertEqual(product.output_type, "VISUAL_RGB")
        self.assertEqual(product.collection, "test_collection")
        self.assertEqual(product.stac_datetime, "2024-01-01T00:00:00Z")


class TestEdcLineItem(unittest.TestCase):
    def test_from_dict_full(self):
        data = {
            "id": "item_id",
            "order_id": "order_id",
            "order_name": "order_name",
            "tracking_id": "tracking_id",
            "state": "COMPLETED",
            "input_uuid": "input_uuid",
            "product_type": "VISUAL_RGB",
            "created_date": "2024-01-01T00:00:00Z",
            "account_id": "account_id",
            "user_id": "user_id",
            "satellite": "EarthDaily03",
            "input_type": "L1C",
            "output_type": "VISUAL_RGB",
            "input_namespace": "simedc",
            "latency_tier": "standard",
            "product_points": 100,
            "sqkm": 100.5,
            "collection": "test_collection",
            "type": "ARCHIVE",
            "output_uuid": "output_uuid",
            "canonical_order_id": "canonical_id",
            "processed_date": "2024-01-01T00:00:00Z",
            "aoi": {"type": "Point"},
            "processed_products": [
                {
                    "processed_product_uuid": "prod_uuid",
                    "state": "COMPLETED",
                    "processed_date": "2024-01-01T00:00:00Z",
                    "sqkm": 100.5,
                    "output_type": "VISUAL_RGB",
                    "collection": "test_collection",
                    "stac_datetime": "2024-01-01T00:00:00Z",
                }
            ],
            "input_query": {"field": "value"},
            "resubmitted_id": "resubmitted_id",
            "is_ecommerce": True,
            "payment_intent_id": "payment_intent_id",
            "core_execution_arn": "arn:aws:states:us-east-1:123456789012:execution:test",
        }

        line_item = EdcLineItem.from_dict(data)

        self.assertEqual(line_item.id, "item_id")
        self.assertEqual(line_item.order_id, "order_id")
        self.assertEqual(line_item.state, "COMPLETED")
        self.assertEqual(line_item.output_uuid, "output_uuid")
        self.assertEqual(len(line_item.processed_products), 1)
        self.assertTrue(line_item.is_ecommerce)
        self.assertEqual(line_item.payment_intent_id, "payment_intent_id")

    def test_from_dict_minimal(self):
        data = {
            "id": "item_id",
            "order_id": "order_id",
            "order_name": "order_name",
            "tracking_id": "tracking_id",
            "state": "IN_PROGRESS",
            "input_uuid": "input_uuid",
            "product_type": "VISUAL_RGB",
            "created_date": "2024-01-01T00:00:00Z",
            "account_id": "account_id",
            "user_id": "user_id",
            "satellite": "EarthDaily03",
            "input_type": "L1C",
            "output_type": "VISUAL_RGB",
            "input_namespace": "simedc",
            "latency_tier": "standard",
            "product_points": 100,
            "sqkm": 100.5,
            "collection": None,
            "type": "ARCHIVE",
        }

        line_item = EdcLineItem.from_dict(data)

        self.assertEqual(line_item.id, "item_id")
        self.assertIsNone(line_item.output_uuid)
        self.assertIsNone(line_item.canonical_order_id)
        self.assertIsNone(line_item.processed_date)
        self.assertIsNone(line_item.aoi)
        self.assertEqual(len(line_item.processed_products), 0)
        self.assertFalse(line_item.is_ecommerce)


class TestEdcLineItemResponse(unittest.TestCase):
    def test_from_dict_with_output(self):
        data = {
            "id": "item_id",
            "order_id": "order_id",
            "order_name": "order_name",
            "tracking_id": "tracking_id",
            "state": "COMPLETED",
            "input_uuid": "input_uuid",
            "product_type": "VISUAL_RGB",
            "created_date": "2024-01-01T00:00:00Z",
            "output_uuid": "output_uuid",
        }

        response = EdcLineItemResponse.from_dict(data)

        self.assertEqual(response.id, "item_id")
        self.assertEqual(response.output_uuid, "output_uuid")

    def test_from_dict_without_output(self):
        data = {
            "id": "item_id",
            "order_id": "order_id",
            "order_name": "order_name",
            "tracking_id": "tracking_id",
            "state": "IN_PROGRESS",
            "input_uuid": "input_uuid",
            "product_type": "VISUAL_RGB",
            "created_date": "2024-01-01T00:00:00Z",
        }

        response = EdcLineItemResponse.from_dict(data)

        self.assertEqual(response.id, "item_id")
        self.assertIsNone(response.output_uuid)


class TestEdcOrder(unittest.TestCase):
    def test_from_dict_with_line_items(self):
        data = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
            "uuid": "order_uuid",
            "criteria": {"field": "value"},
            "line_items": [
                {
                    "id": "item_id",
                    "order_id": "order_id",
                    "order_name": "order_name",
                    "tracking_id": "tracking_id",
                    "state": "COMPLETED",
                    "input_uuid": "input_uuid",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                }
            ],
        }

        order = EdcOrder.from_dict(data)

        self.assertEqual(order.id, "order_id")
        self.assertEqual(order.uuid, "order_uuid")
        self.assertEqual(len(order.line_items), 1)
        self.assertEqual(order.line_items[0].id, "item_id")

    def test_from_dict_without_line_items(self):
        data = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
        }

        order = EdcOrder.from_dict(data)

        self.assertEqual(order.id, "order_id")
        self.assertIsNone(order.uuid)
        self.assertIsNone(order.criteria)
        self.assertIsNone(order.line_items)


class TestEdcOrdersService(unittest.TestCase):
    def setUp(self):
        self.mock_api_requester = Mock(spec=APIRequester)
        self.mock_api_requester.base_url = "https://api.earthdaily.com"
        self.service = EdcOrdersService(self.mock_api_requester)

    def test_init(self):
        self.assertEqual(self.service.api_requester, self.mock_api_requester)

    def test_generate_random_suffix_default_length(self):
        suffix = self.service._generate_random_suffix()
        self.assertEqual(len(suffix), 7)
        self.assertTrue(suffix.isalnum())
        self.assertTrue(suffix.islower())

    def test_generate_random_suffix_custom_length(self):
        suffix = self.service._generate_random_suffix(10)
        self.assertEqual(len(suffix), 10)

    def test_generate_order_name_without_suffix(self):
        order_name = self.service._generate_order_name("VISUAL_RGB")
        self.assertTrue(order_name.startswith("eda_client_"))
        self.assertIn("visual_rgb", order_name)
        parts = order_name.split("_")
        self.assertEqual(len(parts[-1]), 7)

    def test_generate_order_name_with_custom_suffix(self):
        order_name = self.service._generate_order_name("VISUAL_RGB", "custom123")
        self.assertTrue(order_name.startswith("eda_client_"))
        self.assertTrue(order_name.endswith("custom123"))
        self.assertIn("visual_rgb", order_name)

    def test_send_request_success(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 200
        mock_response.body = {"result": "success"}
        self.mock_api_requester.send_request.return_value = mock_response

        result = self.service._send_request("GET", "test/endpoint")

        self.assertEqual(result, {"result": "success"})
        self.mock_api_requester.send_request.assert_called_once()

    def test_send_request_success_201(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 201
        mock_response.body = {"result": "created"}
        self.mock_api_requester.send_request.return_value = mock_response

        result = self.service._send_request("POST", "test/endpoint", {"data": "value"})

        self.assertEqual(result, {"result": "created"})

    def test_send_request_failure(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 400
        mock_response.body = {"error": "bad request"}
        self.mock_api_requester.send_request.return_value = mock_response

        with self.assertRaises(EDSAPIError) as context:
            self.service._send_request("POST", "test/endpoint")

        self.assertIn("400", str(context.exception))

    def test_send_request_500_error(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 500
        mock_response.body = {"error": "internal server error"}
        self.mock_api_requester.send_request.return_value = mock_response

        with self.assertRaises(EDSAPIError) as context:
            self.service._send_request("GET", "test/endpoint")

        self.assertIn("500", str(context.exception))

    def test_get_line_items(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 200
        mock_response.body = {
            "line_items": [
                {
                    "id": "item_id",
                    "order_id": "order_id",
                    "order_name": "order_name",
                    "tracking_id": "tracking_id",
                    "state": "COMPLETED",
                    "input_uuid": "input_uuid",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                    "account_id": "account_id",
                    "user_id": "user_id",
                    "satellite": "EarthDaily03",
                    "input_type": "L1C",
                    "output_type": "VISUAL_RGB",
                    "input_namespace": "simedc",
                    "latency_tier": "standard",
                    "product_points": 100,
                    "sqkm": 100.5,
                    "collection": "test_collection",
                    "type": "ARCHIVE",
                }
            ]
        }
        self.mock_api_requester.send_request.return_value = mock_response

        line_items = self.service.get_line_items("order_id")

        self.assertEqual(len(line_items), 1)
        self.assertIsInstance(line_items[0], EdcLineItem)
        self.assertEqual(line_items[0].id, "item_id")

    def test_create_with_string_list(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 201
        mock_response.body = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
            "line_items": [
                {
                    "id": "item_id",
                    "order_id": "order_id",
                    "order_name": "order_name",
                    "tracking_id": "tracking_id",
                    "state": "IN_PROGRESS",
                    "input_uuid": "item_id_1",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                }
            ],
        }
        self.mock_api_requester.send_request.return_value = mock_response

        order = self.service.create(items=["item_id_1"], product_type=EdcProductType.VISUAL_RGB)

        self.assertIsInstance(order, EdcOrder)
        self.assertEqual(order.id, "order_id")
        self.mock_api_requester.send_request.assert_called_once()

    def test_create_with_multiple_strings(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 201
        mock_response.body = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
            "line_items": [
                {
                    "id": "item_id_1",
                    "order_id": "order_id",
                    "order_name": "order_name_1",
                    "tracking_id": "tracking_id_1",
                    "state": "IN_PROGRESS",
                    "input_uuid": "item_id_1",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "item_id_2",
                    "order_id": "order_id",
                    "order_name": "order_name_2",
                    "tracking_id": "tracking_id_2",
                    "state": "IN_PROGRESS",
                    "input_uuid": "item_id_2",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                },
            ],
        }
        self.mock_api_requester.send_request.return_value = mock_response

        order = self.service.create(items=["item_id_1", "item_id_2"], product_type=EdcProductType.VISUAL_RGB)

        self.assertEqual(len(order.line_items), 2)

    def test_create_with_pystac_item(self):
        mock_item = Mock(spec=pystac.Item)
        mock_item.id = "stac_item_id"

        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 201
        mock_response.body = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
            "line_items": [
                {
                    "id": "item_id",
                    "order_id": "order_id",
                    "order_name": "order_name",
                    "tracking_id": "tracking_id",
                    "state": "IN_PROGRESS",
                    "input_uuid": "stac_item_id",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                }
            ],
        }
        self.mock_api_requester.send_request.return_value = mock_response

        order = self.service.create(items=[mock_item], product_type=EdcProductType.VISUAL_RGB)

        self.assertEqual(order.line_items[0].input_uuid, "stac_item_id")

    def test_create_with_mixed_items(self):
        mock_item = Mock(spec=pystac.Item)
        mock_item.id = "stac_item_id"

        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 201
        mock_response.body = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
            "line_items": [
                {
                    "id": "item_id_1",
                    "order_id": "order_id",
                    "order_name": "order_name_1",
                    "tracking_id": "tracking_id_1",
                    "state": "IN_PROGRESS",
                    "input_uuid": "stac_item_id",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "item_id_2",
                    "order_id": "order_id",
                    "order_name": "order_name_2",
                    "tracking_id": "tracking_id_2",
                    "state": "IN_PROGRESS",
                    "input_uuid": "string_item_id",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                },
            ],
        }
        self.mock_api_requester.send_request.return_value = mock_response

        order = self.service.create(items=[mock_item, "string_item_id"], product_type=EdcProductType.VISUAL_RGB)

        self.assertEqual(len(order.line_items), 2)

    def test_create_with_custom_suffix_single_item(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 201
        mock_response.body = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
            "line_items": [
                {
                    "id": "item_id",
                    "order_id": "order_id",
                    "order_name": "order_name",
                    "tracking_id": "tracking_id",
                    "state": "IN_PROGRESS",
                    "input_uuid": "item_id_1",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                }
            ],
        }
        self.mock_api_requester.send_request.return_value = mock_response

        order = self.service.create(
            items=["item_id_1"], product_type=EdcProductType.VISUAL_RGB, order_name_suffix="custom_suffix"
        )

        self.assertIsInstance(order, EdcOrder)

    def test_create_with_custom_suffix_multiple_items(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 201
        mock_response.body = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
            "line_items": [
                {
                    "id": "item_id_1",
                    "order_id": "order_id",
                    "order_name": "order_name_1",
                    "tracking_id": "tracking_id_1",
                    "state": "IN_PROGRESS",
                    "input_uuid": "item_id_1",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "item_id_2",
                    "order_id": "order_id",
                    "order_name": "order_name_2",
                    "tracking_id": "tracking_id_2",
                    "state": "IN_PROGRESS",
                    "input_uuid": "item_id_2",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                },
            ],
        }
        self.mock_api_requester.send_request.return_value = mock_response

        order = self.service.create(
            items=["item_id_1", "item_id_2"], product_type=EdcProductType.VISUAL_RGB, order_name_suffix="batch"
        )

        self.assertEqual(len(order.line_items), 2)

    def test_create_with_string_product_type(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 201
        mock_response.body = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
            "line_items": [
                {
                    "id": "item_id",
                    "order_id": "order_id",
                    "order_name": "order_name",
                    "tracking_id": "tracking_id",
                    "state": "IN_PROGRESS",
                    "input_uuid": "item_id_1",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                }
            ],
        }
        self.mock_api_requester.send_request.return_value = mock_response

        order = self.service.create(items=["item_id_1"], product_type="visual_rgb")

        self.assertIsInstance(order, EdcOrder)
        call_args = self.mock_api_requester.send_request.call_args
        request_body = call_args[0][0].body
        self.assertEqual(request_body["line_items"][0]["product_type"], "VISUAL_RGB")

    def test_create_with_uppercase_string_product_type(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 201
        mock_response.body = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
            "line_items": [
                {
                    "id": "item_id",
                    "order_id": "order_id",
                    "order_name": "order_name",
                    "tracking_id": "tracking_id",
                    "state": "IN_PROGRESS",
                    "input_uuid": "item_id_1",
                    "product_type": "VISUAL_RGB",
                    "created_date": "2024-01-01T00:00:00Z",
                }
            ],
        }
        self.mock_api_requester.send_request.return_value = mock_response

        order = self.service.create(items=["item_id_1"], product_type="VISUAL_RGB")

        self.assertIsInstance(order, EdcOrder)
        call_args = self.mock_api_requester.send_request.call_args
        request_body = call_args[0][0].body
        self.assertEqual(request_body["line_items"][0]["product_type"], "VISUAL_RGB")

    def test_create_with_unsupported_product_type(self):
        with self.assertRaises(ValueError) as context:
            self.service.create(items=["item_id_1"], product_type="UNSUPPORTED_TYPE")

        self.assertIn("Unsupported product type", str(context.exception))
        self.assertIn("UNSUPPORTED_TYPE", str(context.exception))
        self.assertIn("VISUAL_RGB", str(context.exception))

    def test_get(self):
        mock_response = Mock(spec=HTTPResponse)
        mock_response.status_code = 200
        mock_response.body = {
            "id": "order_id",
            "type": "ARCHIVE",
            "account_id": "account_id",
            "tracking_id": "tracking_id",
            "submission_date": "2024-01-01T00:00:00Z",
            "order_name": "order_name",
        }
        self.mock_api_requester.send_request.return_value = mock_response

        order = self.service.get("order_id")

        self.assertIsInstance(order, EdcOrder)
        self.assertEqual(order.id, "order_id")
        self.mock_api_requester.send_request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
