from __future__ import annotations

import random
import string
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pystac

from earthdaily import __version__
from earthdaily._api_requester import APIRequester
from earthdaily._eds_models import BaseRequest, BaseResponse
from earthdaily._http_client import HTTPRequest
from earthdaily.exceptions import EDSAPIError


class EdcProductType(str, Enum):
    VISUAL_RGB = "VISUAL_RGB"


@dataclass
class EdcOrderLineItem:
    """
    Represents a single line item in an EDC order.

    Attributes:
    -----------
    input_uuid : str
        The STAC item ID from a catalog collection.
    product_type : str
        The product type to order (e.g., "VISUAL_RGB").
    order_name : str
        The name for this order line item.
    """

    input_uuid: str
    product_type: str
    order_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_uuid": self.input_uuid,
            "product_type": self.product_type,
            "order_name": self.order_name,
        }


@dataclass
class EdcOrderRequest(BaseRequest):
    """
    Request model for EDC order operations.

    Attributes:
    -----------
    line_items : list[EdcOrderLineItem]
        List of line items to order.
    """

    line_items: list[EdcOrderLineItem]

    def to_dict(self) -> dict[str, Any]:
        return {"line_items": [item.to_dict() for item in self.line_items]}


@dataclass
class EdcProcessedProduct:
    processed_product_uuid: str
    state: str
    processed_date: str
    sqkm: float
    output_type: str
    collection: str
    stac_datetime: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdcProcessedProduct:
        return cls(
            processed_product_uuid=data["processed_product_uuid"],
            state=data["state"],
            processed_date=data["processed_date"],
            sqkm=data["sqkm"],
            output_type=data["output_type"],
            collection=data["collection"],
            stac_datetime=data["stac_datetime"],
        )


@dataclass
class EdcLineItem:
    id: str
    order_id: str
    order_name: str
    tracking_id: str
    state: str
    input_uuid: str
    product_type: str
    created_date: str
    account_id: str
    user_id: str
    satellite: str
    input_type: str
    output_type: str
    input_namespace: str
    latency_tier: str
    product_points: int
    sqkm: float
    collection: str | None
    type: str
    output_uuid: str | None = None
    canonical_order_id: str | None = None
    processed_date: str | None = None
    aoi: Any | None = None
    processed_products: list[EdcProcessedProduct] | None = None
    input_query: Any | None = None
    resubmitted_id: str | None = None
    is_ecommerce: bool = False
    payment_intent_id: str | None = None
    core_execution_arn: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdcLineItem:
        return cls(
            id=data["id"],
            order_id=data["order_id"],
            order_name=data["order_name"],
            tracking_id=data["tracking_id"],
            state=data["state"],
            input_uuid=data["input_uuid"],
            product_type=data["product_type"],
            created_date=data["created_date"],
            account_id=data["account_id"],
            user_id=data["user_id"],
            satellite=data["satellite"],
            input_type=data["input_type"],
            output_type=data["output_type"],
            input_namespace=data["input_namespace"],
            latency_tier=data["latency_tier"],
            product_points=data["product_points"],
            sqkm=data["sqkm"],
            collection=data.get("collection"),
            type=data["type"],
            output_uuid=data.get("output_uuid"),
            canonical_order_id=data.get("canonical_order_id"),
            processed_date=data.get("processed_date"),
            aoi=data.get("aoi"),
            processed_products=[EdcProcessedProduct.from_dict(p) for p in (data.get("processed_products") or [])],
            input_query=data.get("input_query"),
            resubmitted_id=data.get("resubmitted_id"),
            is_ecommerce=data.get("is_ecommerce", False),
            payment_intent_id=data.get("payment_intent_id"),
            core_execution_arn=data.get("core_execution_arn"),
        )


@dataclass
class EdcLineItemResponse:
    id: str
    order_id: str
    order_name: str
    tracking_id: str
    state: str
    input_uuid: str
    product_type: str
    created_date: str
    output_uuid: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdcLineItemResponse:
        return cls(
            id=data["id"],
            order_id=data["order_id"],
            order_name=data["order_name"],
            tracking_id=data["tracking_id"],
            state=data["state"],
            input_uuid=data["input_uuid"],
            product_type=data["product_type"],
            created_date=data["created_date"],
            output_uuid=data.get("output_uuid"),
        )


@dataclass
class EdcOrder(BaseResponse):
    id: str
    type: str
    account_id: str
    tracking_id: str
    submission_date: str
    order_name: str
    uuid: str | None = None
    criteria: Any | None = None
    line_items: list[EdcLineItemResponse] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdcOrder:
        return cls(
            id=data["id"],
            type=data["type"],
            account_id=data["account_id"],
            tracking_id=data["tracking_id"],
            submission_date=data["submission_date"],
            order_name=data["order_name"],
            uuid=data.get("uuid"),
            criteria=data.get("criteria"),
            line_items=(
                [EdcLineItemResponse.from_dict(item) for item in data["line_items"]]
                if "line_items" in data and data["line_items"] is not None
                else None
            ),
        )


class EdcOrdersService:
    """
    Service for interacting with EDC orders endpoint.

    Attributes:
    -----------
    api_requester : APIRequester
        An instance of APIRequester used to send HTTP requests to the EDS API.
    """

    def __init__(self, api_requester: APIRequester):
        """
        Initialize the EdcOrdersService.

        Parameters:
        -----------
        api_requester : APIRequester
            An instance of APIRequester used to send HTTP requests to the EDS API.
        """
        self.api_requester = api_requester

    def _generate_random_suffix(self, length: int = 7) -> str:
        """
        Generate a random alphanumeric string.

        Parameters:
        -----------
        length : int
            Length of the random string. Defaults to 7.

        Returns:
        --------
        str
            Random alphanumeric string.
        """
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def _generate_order_name(self, product_type: str, custom_suffix: str | None = None) -> str:
        """
        Generate an order name following the pattern:
        eda_client_{version}_{product_type}_{suffix}

        Parameters:
        -----------
        product_type : str
            The product type being ordered.
        custom_suffix : str | None
            Custom suffix to use. If None, generates a random 7-character string.

        Returns:
        --------
        str
            The generated order name.
        """
        version_str = __version__.replace(".", "_")
        suffix = custom_suffix if custom_suffix else self._generate_random_suffix()
        return f"eda_client_{version_str}_{product_type.lower()}_{suffix}"

    def _send_request(self, method: str, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Send a request to an ordering endpoint.

        Parameters:
        -----------
        method : str
            HTTP method for the request (e.g., "GET", "POST").
        endpoint : str
            The specific endpoint to send the request to.
        data : dict[str, Any] | None, optional
            The payload to be sent in the request.

        Returns:
        --------
        dict[str, Any]
            The response from the ordering API.

        Raises:
        -------
        EDSAPIError
            If the API returns an error response.
        """
        url = f"{self.api_requester.base_url}/{endpoint}"
        request = HTTPRequest(method=method, url=url, body=data)
        response = self.api_requester.send_request(request)

        if response.status_code not in [200, 201]:
            raise EDSAPIError(f"API request failed with status {response.status_code}: {response.body}")

        return response.body

    def get_line_items(self, order_id: str) -> list[EdcLineItem]:
        response = self._send_request("POST", "ordering/v3/line_items/query", {"order_id": order_id})
        return [EdcLineItem.from_dict(item) for item in response["line_items"]]

    def create(
        self,
        items: list[pystac.Item | str],
        product_type: EdcProductType | str,
        order_name_suffix: str | None = None,
    ) -> EdcOrder:
        """
        Create an EDC order.

        Parameters:
        -----------
        items : list[pystac.Item | str]
            List of items to order. Can be pystac.Item objects or item IDs as strings.
        product_type : EdcProductType | str
            The product type to order. Can be an EdcProductType enum value or a string.
            String values will be converted to uppercase.
        order_name_suffix : str | None
            Custom suffix for the order name. If None, a random 7-character string is generated for each item.
            If provided, each item will get the suffix with an incremental number appended (e.g., suffix_1, suffix_2).

        Returns:
        --------
        EdcOrder
            The EdcOrder object containing order details.

        Examples:
        ---------
        >>> order = service.create(["item_id_123"], EdcProductType.VISUAL_RGB)
        >>> order = service.create(["item_1", "item_2"], "VISUAL_RGB", "custom_suffix")
        >>> order = service.create([pystac_item], "visual_rgb")
        """
        if isinstance(product_type, EdcProductType):
            product_type_str = product_type.value
        else:
            product_type_str = product_type.upper()
            valid_types = [pt.value for pt in EdcProductType]
            if product_type_str not in valid_types:
                raise ValueError(f"Unsupported product type: {product_type}. Supported types: {', '.join(valid_types)}")

        line_items = []
        for idx, item in enumerate(items, start=1):
            if isinstance(item, pystac.Item):
                item_id = item.id
            else:
                item_id = item

            if order_name_suffix:
                suffix = f"{order_name_suffix}_{idx}" if len(items) > 1 else order_name_suffix
            else:
                suffix = None

            order_name = self._generate_order_name(product_type_str, suffix)
            line_items.append(
                EdcOrderLineItem(input_uuid=item_id, product_type=product_type_str, order_name=order_name)
            )

        request_data = EdcOrderRequest(line_items=line_items)
        response = self._send_request("POST", "ordering/v3/orders", request_data.to_dict())
        return EdcOrder.from_dict(response)

    def get(self, order_id: str) -> EdcOrder:
        """
        Get an EDC order by its ID.

        Parameters:
        -----------
        order_id : str
            The ID of the order to retrieve.

        Returns:
        --------
        EdcOrder
            The EdcOrder object.

        Examples:
        ---------
        >>> order = service.get("order_id_123")
        """
        response = self._send_request("GET", f"ordering/v3/orders/{order_id}")
        return EdcOrder.from_dict(response)
