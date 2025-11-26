from earthdaily._api_requester import APIRequester
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

__all__ = [
    "OrderingService",
    "EdcOrdersService",
    "EdcOrderRequest",
    "EdcOrder",
    "EdcOrderLineItem",
    "EdcLineItem",
    "EdcLineItemResponse",
    "EdcProcessedProduct",
    "EdcProductType",
]


class OrderingService:
    """
    Represents the Ordering Service for interacting with ordering-related endpoints.

    Attributes:
    -----------
    api_requester : APIRequester
        An instance of APIRequester used to send HTTP requests to the EDS API.
    """

    def __init__(self, api_requester: APIRequester):
        """
        Initialize the OrderingService.

        Parameters:
        -----------
        api_requester : APIRequester
            An instance of APIRequester used to send HTTP requests to the EDS API.
        """
        self.api_requester = api_requester
        self.edc = EdcOrdersService(api_requester)
