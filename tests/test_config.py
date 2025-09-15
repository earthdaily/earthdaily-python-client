import json
import os
import tempfile
import unittest

from earthdaily import EDSConfig
from earthdaily._eds_config import AssetAccessMode


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

    def test_default_base_url(self):
        os.environ["EDS_CLIENT_ID"] = "env_client_id"
        os.environ["EDS_SECRET"] = "env_client_secret"
        os.environ["EDS_AUTH_URL"] = "env_token_url"

        config = EDSConfig()

        self.assertEqual(config.base_url, "https://api.earthdaily.com")

    def test_bypass_auth(self):
        config = EDSConfig(bypass_auth=True)

        self.assertTrue(config.bypass_auth)

    def test_asset_access_mode_default(self):
        os.environ["EDS_CLIENT_ID"] = "env_client_id"
        os.environ["EDS_SECRET"] = "env_client_secret"
        os.environ["EDS_AUTH_URL"] = "env_token_url"

        config = EDSConfig()

        self.assertEqual(config.asset_access_mode, AssetAccessMode.PRESIGNED_URLS)

    def test_asset_access_mode_custom(self):
        os.environ["EDS_CLIENT_ID"] = "env_client_id"
        os.environ["EDS_SECRET"] = "env_client_secret"
        os.environ["EDS_AUTH_URL"] = "env_token_url"

        config = EDSConfig(asset_access_mode=AssetAccessMode.RAW)

        self.assertEqual(config.asset_access_mode, AssetAccessMode.RAW)

    def test_retry_config_defaults(self):
        os.environ["EDS_CLIENT_ID"] = "env_client_id"
        os.environ["EDS_SECRET"] = "env_client_secret"
        os.environ["EDS_AUTH_URL"] = "env_token_url"

        config = EDSConfig()

        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.retry_backoff_factor, 1.0)

    def test_retry_config_custom(self):
        os.environ["EDS_CLIENT_ID"] = "env_client_id"
        os.environ["EDS_SECRET"] = "env_client_secret"
        os.environ["EDS_AUTH_URL"] = "env_token_url"

        config = EDSConfig(max_retries=5, retry_backoff_factor=2.0)

        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.retry_backoff_factor, 2.0)

    def test_load_from_json_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "EDS_CLIENT_ID": "json_client_id",
                    "EDS_SECRET": "json_client_secret",
                    "EDS_AUTH_URL": "json_token_url",
                },
                f,
            )

        try:
            config = EDSConfig(json_path=f.name)

            self.assertEqual(config.client_id, "json_client_id")
            self.assertEqual(config.client_secret, "json_client_secret")
            self.assertEqual(config.token_url, "json_token_url")
        finally:
            os.unlink(f.name)

    def test_load_from_toml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            toml_content = """
            EDS_CLIENT_ID = "toml_client_id"
            EDS_SECRET = "toml_client_secret"
            EDS_AUTH_URL = "toml_token_url"
            """
            f.write(toml_content)

        try:
            config = EDSConfig(toml_path=f.name)

            self.assertEqual(config.client_id, "toml_client_id")
            self.assertEqual(config.client_secret, "toml_client_secret")
            self.assertEqual(config.token_url, "toml_token_url")
        finally:
            os.unlink(f.name)

    def test_load_from_ini_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            ini_content = """
            [default]
            eds_client_id = ini_client_id
            eds_secret = ini_client_secret
            eds_auth_url = ini_token_url
            
            [other_profile]
            eds_client_id = other_client_id
            eds_secret = other_client_secret
            eds_auth_url = other_token_url
            """
            f.write(ini_content)

        try:
            config = EDSConfig(ini_path=f.name)

            self.assertEqual(config.client_id, "ini_client_id")
            self.assertEqual(config.client_secret, "ini_client_secret")
            self.assertEqual(config.token_url, "ini_token_url")

            config = EDSConfig(ini_path=f.name, ini_profile="other_profile")

            self.assertEqual(config.client_id, "other_client_id")
            self.assertEqual(config.client_secret, "other_client_secret")
            self.assertEqual(config.token_url, "other_token_url")
        finally:
            os.unlink(f.name)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            EDSConfig(json_path="/nonexistent/path.json")

    def test_unsupported_file_type(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")

        try:
            with self.assertRaises(ValueError) as context:
                EDSConfig(json_path=f.name)

            self.assertIn("Unsupported config file type", str(context.exception))
        finally:
            os.unlink(f.name)

    def test_invalid_ini_profile(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            ini_content = """
            [default]
            eds_client_id = ini_client_id
            eds_secret = ini_client_secret
            eds_auth_url = ini_token_url
            """
            f.write(ini_content)

        try:
            with self.assertRaises(ValueError) as context:
                EDSConfig(ini_path=f.name, ini_profile="nonexistent_profile")

            self.assertIn("Profile 'nonexistent_profile' not found", str(context.exception))
        finally:
            os.unlink(f.name)

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

    def test_config_priority_order(self):
        # Set environment variables
        os.environ["EDS_CLIENT_ID"] = "env_client_id"
        os.environ["EDS_SECRET"] = "env_client_secret"
        os.environ["EDS_AUTH_URL"] = "env_token_url"

        # Create config with direct parameters (should take precedence)
        config = EDSConfig(
            client_id="direct_client_id", client_secret="direct_client_secret", token_url="direct_token_url"
        )

        # Direct parameters should take precedence
        self.assertEqual(config.client_id, "direct_client_id")
        self.assertEqual(config.client_secret, "direct_client_secret")
        self.assertEqual(config.token_url, "direct_token_url")

    def test_ini_env_var_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            ini_content = """
            [default]
            eds_client_id = ini_client_id
            eds_secret = ini_client_secret
            eds_auth_url = ini_token_url
            """
            f.write(ini_content)

        try:
            # Set environment variable to point to the ini file
            os.environ["EDS_INI_PATH"] = f.name
            config = EDSConfig()

            self.assertEqual(config.client_id, "ini_client_id")
            self.assertEqual(config.client_secret, "ini_client_secret")
            self.assertEqual(config.token_url, "ini_token_url")
        finally:
            os.unlink(f.name)

    def test_json_env_var_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "EDS_CLIENT_ID": "json_client_id",
                    "EDS_SECRET": "json_client_secret",
                    "EDS_AUTH_URL": "json_token_url",
                },
                f,
            )

        try:
            # Set environment variable to point to the json file
            os.environ["EDS_JSON_PATH"] = f.name
            config = EDSConfig()

            self.assertEqual(config.client_id, "json_client_id")
            self.assertEqual(config.client_secret, "json_client_secret")
            self.assertEqual(config.token_url, "json_token_url")
        finally:
            os.unlink(f.name)

    def test_file_precedence(self):
        # Create 3 config files with different values
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as json_file,
            tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as toml_file,
            tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as ini_file,
        ):
            # JSON config
            json.dump(
                {
                    "EDS_CLIENT_ID": "json_client_id",
                    "EDS_SECRET": "json_client_secret",
                    "EDS_AUTH_URL": "json_token_url",
                },
                json_file,
            )

            # TOML config
            toml_content = """
            EDS_CLIENT_ID = "toml_client_id"
            EDS_SECRET = "toml_client_secret"
            EDS_AUTH_URL = "toml_token_url"
            """
            toml_file.write(toml_content)

            # INI config
            ini_content = """
            [default]
            eds_client_id = ini_client_id
            eds_secret = ini_client_secret
            eds_auth_url = ini_token_url
            """
            ini_file.write(ini_content)

            # Flush files to ensure content is written
            json_file.flush()
            toml_file.flush()
            ini_file.flush()

            try:
                # Test JSON precedence (should be first)
                config = EDSConfig(json_path=json_file.name, toml_path=toml_file.name, ini_path=ini_file.name)
                self.assertEqual(config.client_id, "json_client_id")

                # Test TOML precedence (when JSON not provided)
                config = EDSConfig(toml_path=toml_file.name, ini_path=ini_file.name)
                self.assertEqual(config.client_id, "toml_client_id")

                # Test INI is used when others not provided
                config = EDSConfig(ini_path=ini_file.name)
                self.assertEqual(config.client_id, "ini_client_id")
            finally:
                os.unlink(json_file.name)
                os.unlink(toml_file.name)
                os.unlink(ini_file.name)


if __name__ == "__main__":
    unittest.main()
