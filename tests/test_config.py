import os
import unittest

from earthdaily import EDSConfig


class TestEDSConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Store original environment
        cls.original_env = dict(os.environ)

    def setUp(self):
        # Clear environment variables before each test
        os.environ.clear()
        os.environ.update(self.original_env)

    def tearDown(self):
        # Restore original environment after each test
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_config_with_env_vars(self):
        os.environ["EDS_CLIENT_ID"] = "env_client_id"
        os.environ["EDS_SECRET"] = "env_client_secret"
        os.environ["EDS_AUTH_URL"] = "env_token_url"
        os.environ["EDS_API_URL"] = "https://EDS_API_URL.com"

        config = EDSConfig()

        self.assertEqual(config.client_id, "env_client_id")
        self.assertEqual(config.client_secret, "env_client_secret")
        self.assertEqual(config.token_url, "env_token_url")
        self.assertEqual(config.base_url, "https://EDS_API_URL.com")

    def test_config_with_passed_values(self):
        config = EDSConfig(
            client_id="input_client_id",
            client_secret="input_client_secret",
            token_url="input_token_url",
        )

        self.assertEqual(config.client_id, "input_client_id")
        self.assertEqual(config.client_secret, "input_client_secret")
        self.assertEqual(config.token_url, "input_token_url")

    def test_config_missing_required_fields(self):
        # EDS_CLIENT_ID is missing
        os.environ["EDS_CLIENT_ID"] = ""
        os.environ["EDS_SECRET"] = "env_client_secret"
        os.environ["EDS_AUTH_URL"] = "env_token_url"
        os.environ["EDS_API_URL"] = "https://EDS_API_URL.com"

        with self.assertRaises(ValueError) as context:
            EDSConfig()

        self.assertIn("Missing required fields", str(context.exception))


if __name__ == "__main__":
    unittest.main()
