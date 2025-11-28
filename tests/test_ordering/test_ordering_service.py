from __future__ import annotations

import unittest
from unittest.mock import Mock

from earthdaily._api_requester import APIRequester
from earthdaily.ordering import EdcOrdersService, OrderingService


class TestOrderingService(unittest.TestCase):
    def test_init(self):
        mock_api_requester = Mock(spec=APIRequester)
        service = OrderingService(mock_api_requester)

        self.assertEqual(service.api_requester, mock_api_requester)
        self.assertIsInstance(service.edc, EdcOrdersService)
        self.assertEqual(service.edc.api_requester, mock_api_requester)


if __name__ == "__main__":
    unittest.main()
