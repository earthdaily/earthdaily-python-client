import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

from pystac import Item, ItemCollection

from earthdaily._api_requester import APIRequester
from earthdaily._eds_config import AssetAccessMode
from earthdaily.platform import PlatformService


class TestPlatformSearch(unittest.TestCase):
    def setUp(self):
        self.mock_api_requester = Mock(spec=APIRequester)
        self.mock_api_requester.base_url = "https://api.example.com"
        self.mock_api_requester.headers = {}
        self.mock_api_requester.auth = None
        self.mock_api_requester.config = Mock()
        self.mock_api_requester.config.max_retries = 3
        self.mock_api_requester.config.retry_backoff_factor = 0.5

    @patch("earthdaily.platform.Client.open")
    def test_search_without_days_per_chunk_uses_pystac_client(self, mock_client_open):
        mock_client = MagicMock()
        mock_items = [self._create_mock_item("item1"), self._create_mock_item("item2")]
        mock_client.search.return_value.item_collection.return_value = ItemCollection(mock_items)
        mock_client_open.return_value = mock_client

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        result = service.search(
            collections=["sentinel-2-l2a"],
            datetime="2023-01-01/2023-06-30",
            bbox=[1, 43, 2, 44],
        )

        mock_client.search.assert_called_once_with(
            collections=["sentinel-2-l2a"],
            datetime="2023-01-01/2023-06-30",
            bbox=[1, 43, 2, 44],
        )
        self.assertEqual(len(result), 2)

    @patch("earthdaily.platform.Client.open")
    def test_search_with_days_per_chunk_none_uses_pystac_client(self, mock_client_open):
        mock_client = MagicMock()
        mock_items = [self._create_mock_item("item1")]
        mock_client.search.return_value.item_collection.return_value = ItemCollection(mock_items)
        mock_client_open.return_value = mock_client

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        result = service.search(
            collections=["sentinel-2-l2a"],
            datetime="2023-01-01/2023-06-30",
            days_per_chunk=None,
        )

        mock_client.search.assert_called_once()
        self.assertEqual(len(result), 1)

    @patch("earthdaily.platform.Client.open")
    def test_search_without_datetime_uses_pystac_client(self, mock_client_open):
        mock_client = MagicMock()
        mock_items = [self._create_mock_item("item1")]
        mock_client.search.return_value.item_collection.return_value = ItemCollection(mock_items)
        mock_client_open.return_value = mock_client

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        result = service.search(
            collections=["sentinel-2-l2a"],
            bbox=[1, 43, 2, 44],
            days_per_chunk="auto",
        )

        mock_client.search.assert_called_once()
        self.assertEqual(len(result), 1)

    @patch("earthdaily.platform.Client.open")
    def test_search_with_days_per_chunk_auto_executes_concurrent(self, mock_client_open):
        mock_client = MagicMock()
        mock_items = [self._create_mock_item("item1")]
        mock_client.search.return_value.item_collection.return_value = ItemCollection(mock_items)
        mock_client.search.return_value.items.return_value = iter(mock_items)
        mock_client_open.return_value = mock_client

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        result = service.search(
            collections=["sentinel-2-l2a"],
            datetime="2020-01-01/2023-12-31",
            days_per_chunk="auto",
        )

        self.assertIsInstance(result, ItemCollection)
        self.assertGreater(mock_client_open.call_count, 1)

    @patch("earthdaily.platform.Client.open")
    def test_search_with_days_per_chunk_int_executes_concurrent(self, mock_client_open):
        mock_client = MagicMock()
        mock_items = [self._create_mock_item("item1")]
        mock_client.search.return_value.item_collection.return_value = ItemCollection(mock_items)
        mock_client.search.return_value.items.return_value = iter(mock_items)
        mock_client_open.return_value = mock_client

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        result = service.search(
            collections=["sentinel-2-l2a"],
            datetime="2023-01-01/2023-12-31",
            days_per_chunk=30,
        )

        self.assertIsInstance(result, ItemCollection)
        self.assertGreater(mock_client_open.call_count, 1)

    def _create_mock_item(self, item_id: str) -> Item:
        return Item(
            id=item_id,
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=[0, 0, 1, 1],
            datetime=datetime.now(),
            properties={},
        )


class TestSplitDatetime(unittest.TestCase):
    def setUp(self):
        self.mock_api_requester = Mock(spec=APIRequester)
        self.mock_api_requester.base_url = "https://api.example.com"
        self.mock_api_requester.headers = {}
        self.mock_api_requester.auth = None
        self.mock_api_requester.config = Mock()
        self.mock_api_requester.config.max_retries = 3
        self.mock_api_requester.config.retry_backoff_factor = 0.5

    @patch("earthdaily.platform.Client.open")
    def test_split_datetime_string_format(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        result = service._split_datetime("2023-01-01/2023-12-31", days_per_chunk=90)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 1)
        for date_range in result:
            self.assertIn("/", date_range)

    @patch("earthdaily.platform.Client.open")
    def test_split_datetime_tuple_format(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        result = service._split_datetime(
            (datetime(2023, 1, 1), datetime(2023, 12, 31)),
            days_per_chunk=90,
        )

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 1)

    @patch("earthdaily.platform.Client.open")
    def test_split_datetime_auto_calculates_chunks(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        result = service._split_datetime("2020-01-01/2023-12-31", days_per_chunk="auto")

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 1)
        self.assertLessEqual(len(result), 15)

    @patch("earthdaily.platform.Client.open")
    def test_split_datetime_small_range_returns_single(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        result = service._split_datetime("2023-01-01/2023-01-15", days_per_chunk=30)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "2023-01-01/2023-01-15")

    @patch("earthdaily.platform.Client.open")
    def test_split_datetime_single_date_returns_as_is(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        result = service._split_datetime("2023-01-01", days_per_chunk=30)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "2023-01-01")

    @patch("earthdaily.platform.Client.open")
    def test_split_datetime_zero_days_per_chunk_raises_error(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        with self.assertRaises(ValueError) as context:
            service._split_datetime("2023-01-01/2023-12-31", days_per_chunk=0)

        self.assertIn("days_per_chunk must be a positive integer", str(context.exception))

    @patch("earthdaily.platform.Client.open")
    def test_split_datetime_negative_days_per_chunk_raises_error(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        with self.assertRaises(ValueError) as context:
            service._split_datetime("2023-01-01/2023-12-31", days_per_chunk=-5)

        self.assertIn("days_per_chunk must be a positive integer", str(context.exception))

    @patch("earthdaily.platform.Client.open")
    def test_split_datetime_open_ended_start_raises_error(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        with self.assertRaises(ValueError) as context:
            service._split_datetime("../2023-12-31", days_per_chunk=30)

        self.assertIn("days_per_chunk cannot be used with open-ended datetime intervals", str(context.exception))

    @patch("earthdaily.platform.Client.open")
    def test_split_datetime_open_ended_end_raises_error(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        with self.assertRaises(ValueError) as context:
            service._split_datetime("2023-01-01/..", days_per_chunk=30)

        self.assertIn("days_per_chunk cannot be used with open-ended datetime intervals", str(context.exception))


class TestCreatePystacClient(unittest.TestCase):
    def setUp(self):
        self.mock_api_requester = Mock(spec=APIRequester)
        self.mock_api_requester.base_url = "https://api.example.com"
        self.mock_api_requester.headers = {"Custom-Header": "value"}
        self.mock_api_requester.auth = Mock()
        self.mock_api_requester.auth.get_token.return_value = "mock_token"
        self.mock_api_requester.config = Mock()
        self.mock_api_requester.config.max_retries = 3
        self.mock_api_requester.config.retry_backoff_factor = 0.5

    @patch("earthdaily.platform.Client.open")
    def test_create_pystac_client_returns_new_instance(self, mock_client_open):
        mock_client_1 = MagicMock()
        mock_client_2 = MagicMock()
        mock_client_3 = MagicMock()
        mock_client_open.side_effect = [mock_client_1, mock_client_2, mock_client_3]

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        client1 = service._create_pystac_client()
        client2 = service._create_pystac_client()

        self.assertIsNot(client1, client2)
        self.assertEqual(mock_client_open.call_count, 3)

    @patch("earthdaily.platform.Client.open")
    def test_create_pystac_client_uses_correct_url(self, mock_client_open):
        mock_client_open.return_value = MagicMock()

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        service._create_pystac_client()

        call_args = mock_client_open.call_args
        self.assertEqual(call_args[0][0], "https://api.example.com/platform/v1/stac")

    @patch("earthdaily.platform.Client.open")
    def test_create_pystac_client_includes_auth_header(self, mock_client_open):
        mock_client_open.return_value = MagicMock()

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        service._create_pystac_client()

        call_args = mock_client_open.call_args
        headers = call_args[1]["headers"]
        self.assertIn("Authorization", headers)
        self.assertEqual(headers["Authorization"], "Bearer mock_token")

    @patch("earthdaily.platform.Client.open")
    def test_create_pystac_client_fetches_fresh_token_each_call(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        tokens = ["token_1", "token_2", "token_3"]
        self.mock_api_requester.auth.get_token.side_effect = tokens

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        service._create_pystac_client()
        service._create_pystac_client()

        self.assertEqual(self.mock_api_requester.auth.get_token.call_count, 3)
        call_args_list = mock_client_open.call_args_list
        self.assertEqual(call_args_list[0][1]["headers"]["Authorization"], "Bearer token_1")
        self.assertEqual(call_args_list[1][1]["headers"]["Authorization"], "Bearer token_2")
        self.assertEqual(call_args_list[2][1]["headers"]["Authorization"], "Bearer token_3")


class TestConcurrentSearchThreadSafety(unittest.TestCase):
    def setUp(self):
        self.mock_api_requester = Mock(spec=APIRequester)
        self.mock_api_requester.base_url = "https://api.example.com"
        self.mock_api_requester.headers = {}
        self.mock_api_requester.auth = None
        self.mock_api_requester.config = Mock()
        self.mock_api_requester.config.max_retries = 3
        self.mock_api_requester.config.retry_backoff_factor = 0.5

    @patch("earthdaily.platform.Client.open")
    def test_concurrent_search_creates_separate_clients(self, mock_client_open):
        clients_created = []

        def create_client(*args, **kwargs):
            client = MagicMock()
            client.search.return_value.items.return_value = iter([])
            clients_created.append(client)
            return client

        mock_client_open.side_effect = create_client

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        service.search(
            collections=["sentinel-2-l2a"],
            datetime="2020-01-01/2023-12-31",
            days_per_chunk=365,
            max_workers=4,
        )

        self.assertGreater(len(clients_created), 1)
        unique_clients = set(id(c) for c in clients_created)
        self.assertEqual(len(unique_clients), len(clients_created))

    @patch("earthdaily.platform.ThreadPoolExecutor")
    @patch("earthdaily.platform.Client.open")
    def test_max_workers_capped_at_10(self, mock_client_open, mock_executor_class):
        mock_client = MagicMock()
        mock_client.search.return_value.items.return_value = iter([])
        mock_client_open.return_value = mock_client

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.return_value = iter([])
        mock_executor_class.return_value = mock_executor

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        service.search(
            collections=["sentinel-2-l2a"],
            datetime="2020-01-01/2023-12-31",
            days_per_chunk=30,
            max_workers=50,
        )

        mock_executor_class.assert_called_once()
        actual_max_workers = mock_executor_class.call_args[1]["max_workers"]
        self.assertLessEqual(actual_max_workers, 10)

    @patch("earthdaily.platform.Client.open")
    def test_max_items_with_days_per_chunk_raises_error(self, mock_client_open):
        mock_client = MagicMock()
        mock_client_open.return_value = mock_client

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        with self.assertRaises(ValueError) as context:
            service.search(
                collections=["sentinel-2-l2a"],
                datetime="2020-01-01/2020-12-31",
                days_per_chunk=30,
                max_items=50,
            )

        self.assertIn("max_items cannot be used with concurrent search", str(context.exception))
        self.assertIn("max_items_per_chunk", str(context.exception))

    @patch("earthdaily.platform.Client.open")
    def test_max_workers_zero_raises_error(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        with self.assertRaises(ValueError) as context:
            service.search(
                collections=["sentinel-2-l2a"],
                datetime="2020-01-01/2020-12-31",
                days_per_chunk=30,
                max_workers=0,
            )

        self.assertIn("max_workers must be a positive integer", str(context.exception))

    @patch("earthdaily.platform.Client.open")
    def test_max_workers_negative_raises_error(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        with self.assertRaises(ValueError) as context:
            service.search(
                collections=["sentinel-2-l2a"],
                datetime="2020-01-01/2020-12-31",
                days_per_chunk=30,
                max_workers=-5,
            )

        self.assertIn("max_workers must be a positive integer", str(context.exception))

    @patch("earthdaily.platform.Client.open")
    def test_max_items_per_chunk_passed_to_chunk_searches(self, mock_client_open):
        chunk_search_kwargs = []

        def create_mock_client():
            client = MagicMock()

            def capture_search(**kwargs):
                chunk_search_kwargs.append(kwargs)
                mock_result = MagicMock()
                mock_result.items.return_value = iter([])
                return mock_result

            client.search.side_effect = capture_search
            return client

        mock_client_open.side_effect = lambda *args, **kwargs: create_mock_client()

        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)
        service.search(
            collections=["sentinel-2-l2a"],
            datetime="2020-01-01/2020-06-30",
            days_per_chunk=30,
            max_items_per_chunk=100,
        )

        for kwargs in chunk_search_kwargs:
            self.assertEqual(kwargs.get("max_items"), 100)

    @patch("earthdaily.platform.Client.open")
    def test_split_datetime_creates_non_overlapping_chunks(self, mock_client_open):
        mock_client_open.return_value = MagicMock()
        service = PlatformService(self.mock_api_requester, AssetAccessMode.PRESIGNED_URLS)

        result = service._split_datetime("2023-01-01/2023-01-21", days_per_chunk=10)

        self.assertEqual(len(result), 2)

        from datetime import datetime as dt

        chunk_ranges = []
        for chunk in result:
            start_str, end_str = chunk.split("/")
            start = dt.fromisoformat(start_str)
            end = dt.fromisoformat(end_str)
            chunk_ranges.append((start, end))

        chunk_ranges.sort(key=lambda x: x[0])
        for i in range(len(chunk_ranges) - 1):
            current_end = chunk_ranges[i][1]
            next_start = chunk_ranges[i + 1][0]
            self.assertLess(current_end, next_start, "Chunks should not overlap")


if __name__ == "__main__":
    unittest.main()
