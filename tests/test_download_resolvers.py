import base64
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from earthdaily.platform._download_resolvers import (
    _EARTHDATA_DOMAINS,
    _S3_HTTPS_PATTERN,
    EarthDataResolver,
    EUMETSATResolver,
    S3Resolver,
)

# Helpers


def _make_jwt(payload: dict, header: dict | None = None) -> str:
    """Build a fake three-part JWT from *payload*."""
    header = header or {"alg": "HS256", "typ": "JWT"}
    h = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{h}.{p}.fake-signature"


def _valid_jwt(exp_offset: int = 3600) -> str:
    return _make_jwt({"exp": int(time.time()) + exp_offset, "sub": "test"})


def _expired_jwt() -> str:
    return _make_jwt({"exp": int(time.time()) - 100, "sub": "test"})


# EarthDataResolver


class TestEarthDataResolverCanHandle(unittest.TestCase):
    def setUp(self):
        self.resolver = EarthDataResolver(token=_valid_jwt())

    def test_handles_all_known_domains(self):
        for domain in _EARTHDATA_DOMAINS:
            url = f"https://{domain}/path/file.hdf"
            self.assertTrue(self.resolver.can_handle(url), f"should handle {domain}")

    def test_rejects_unknown_domain(self):
        assert not self.resolver.can_handle("https://example.com/file.hdf")


class TestEarthDataResolverJWT(unittest.TestCase):
    def test_valid_jwt_passes(self):
        payload = EarthDataResolver._validate_jwt(_valid_jwt())
        assert "exp" in payload

    def test_expired_jwt_raises(self):
        with self.assertRaises(ValueError, msg="expired"):
            EarthDataResolver._validate_jwt(_expired_jwt())

    def test_bad_format_two_parts(self):
        with self.assertRaises(ValueError, msg="three dot-separated"):
            EarthDataResolver._validate_jwt("only.two")

    def test_bad_base64_payload(self):
        with self.assertRaises(ValueError, msg="not valid Base64"):
            EarthDataResolver._validate_jwt("a.!!!invalid!!!.c")

    def test_bad_json_payload(self):
        not_json = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode()
        with self.assertRaises(ValueError, msg="not valid JSON"):
            EarthDataResolver._validate_jwt(f"a.{not_json}.c")

    def test_missing_exp_claim(self):
        token = _make_jwt({"sub": "test"})
        with self.assertRaises(ValueError, msg="missing 'exp'"):
            EarthDataResolver._validate_jwt(token)

    def test_add_base64_padding(self):
        assert EarthDataResolver._add_base64_padding("ab") == "ab=="
        assert EarthDataResolver._add_base64_padding("abc") == "abc="
        assert EarthDataResolver._add_base64_padding("abcd") == "abcd"


class TestEarthDataResolverHeaders(unittest.TestCase):
    def test_valid_token_returns_bearer(self):
        token = _valid_jwt()
        r = EarthDataResolver(token=token)
        headers = r.get_headers("https://data.lpdaac.earthdatacloud.nasa.gov/f")
        assert headers == {"Authorization": f"Bearer {token}"}

    def test_expired_token_raises(self):
        r = EarthDataResolver(token=_expired_jwt())
        with self.assertRaises(ValueError):
            r.get_headers("https://data.lpdaac.earthdatacloud.nasa.gov/f")


class TestEarthDataResolverDownloadUrl(unittest.TestCase):
    def test_passthrough(self):
        r = EarthDataResolver(token=_valid_jwt())
        url = "https://data.lpdaac.earthdatacloud.nasa.gov/file.hdf"
        assert r.get_download_url(url) == url


class TestEarthDataResolverFromConfig(unittest.TestCase):
    def test_returns_instance_when_token_present(self):
        config = SimpleNamespace(earthdata_token="jwt-here")
        r = EarthDataResolver.from_config(config)
        assert isinstance(r, EarthDataResolver)
        assert r._raw_token == "jwt-here"

    def test_returns_none_when_empty(self):
        assert EarthDataResolver.from_config(SimpleNamespace(earthdata_token="")) is None

    def test_returns_none_when_missing_attr(self):
        assert EarthDataResolver.from_config(SimpleNamespace()) is None


# EUMETSATResolver


class TestEUMETSATResolverCanHandle(unittest.TestCase):
    def setUp(self):
        self.resolver = EUMETSATResolver(client_id="id", client_secret="secret")

    def test_handles_eumetsat_domain(self):
        assert self.resolver.can_handle("https://api.eumetsat.int/data/download/file")

    def test_rejects_other(self):
        assert not self.resolver.can_handle("https://example.com/file")


class TestEUMETSATResolverFromConfig(unittest.TestCase):
    def test_returns_instance_when_both_present(self):
        config = SimpleNamespace(eumetsat_client_id="id", eumetsat_client_secret="secret")
        r = EUMETSATResolver.from_config(config)
        assert isinstance(r, EUMETSATResolver)

    def test_returns_none_when_id_missing(self):
        assert EUMETSATResolver.from_config(SimpleNamespace(eumetsat_client_id="", eumetsat_client_secret="s")) is None

    def test_returns_none_when_secret_missing(self):
        assert EUMETSATResolver.from_config(SimpleNamespace(eumetsat_client_id="i", eumetsat_client_secret="")) is None

    def test_returns_none_when_no_attrs(self):
        assert EUMETSATResolver.from_config(SimpleNamespace()) is None


class TestEUMETSATResolverTokenRefresh(unittest.TestCase):
    @patch("earthdaily.platform._download_resolvers.requests.post")
    def test_fetches_token_on_first_call(self, mock_post):
        mock_post.return_value.json.return_value = {
            "access_token": "tok123",
            "expires_in": 3600,
        }
        mock_post.return_value.raise_for_status = MagicMock()

        r = EUMETSATResolver(client_id="cid", client_secret="csec")
        headers = r.get_headers("https://api.eumetsat.int/f")

        assert headers == {"Authorization": "Bearer tok123"}
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.eumetsat.int/token"

    @patch("earthdaily.platform._download_resolvers.requests.post")
    def test_caches_token_on_second_call(self, mock_post):
        mock_post.return_value.json.return_value = {
            "access_token": "tok123",
            "expires_in": 3600,
        }
        mock_post.return_value.raise_for_status = MagicMock()

        r = EUMETSATResolver(client_id="cid", client_secret="csec")
        r.get_headers("https://api.eumetsat.int/f")
        r.get_headers("https://api.eumetsat.int/g")

        mock_post.assert_called_once()

    @patch("earthdaily.platform._download_resolvers.requests.post")
    def test_refreshes_on_expiry(self, mock_post):
        mock_post.return_value.json.return_value = {
            "access_token": "tok_new",
            "expires_in": 3600,
        }
        mock_post.return_value.raise_for_status = MagicMock()

        r = EUMETSATResolver(client_id="cid", client_secret="csec")
        r._bearer_token = "tok_old"
        from datetime import datetime, timedelta, timezone

        r._token_expires = datetime.now(timezone.utc) - timedelta(seconds=1)

        headers = r.get_headers("https://api.eumetsat.int/f")
        assert headers == {"Authorization": "Bearer tok_new"}
        mock_post.assert_called_once()

    @patch("earthdaily.platform._download_resolvers.requests.post")
    def test_raise_on_post_failure(self, mock_post):
        from requests.exceptions import HTTPError

        mock_post.return_value.raise_for_status.side_effect = HTTPError("401")

        r = EUMETSATResolver(client_id="bad", client_secret="bad")
        with self.assertRaises(HTTPError):
            r.get_headers("https://api.eumetsat.int/f")


class TestEUMETSATResolverDownloadUrl(unittest.TestCase):
    def test_passthrough(self):
        r = EUMETSATResolver(client_id="i", client_secret="s")
        url = "https://api.eumetsat.int/data/file"
        assert r.get_download_url(url) == url


# S3Resolver


class TestS3ResolverCanHandle(unittest.TestCase):
    def setUp(self):
        self.resolver = S3Resolver()

    def test_handles_s3_scheme(self):
        assert self.resolver.can_handle("s3://bucket/path/file.tif")

    def test_handles_s3_https(self):
        assert self.resolver.can_handle("https://my-bucket.s3.us-east-1.amazonaws.com/file.tif")

    def test_rejects_s3_directory(self):
        assert not self.resolver.can_handle("s3://bucket/dir/")

    def test_rejects_s3_https_directory(self):
        assert not self.resolver.can_handle("https://my-bucket.s3.us-east-1.amazonaws.com/dir/")

    def test_rejects_plain_http(self):
        assert not self.resolver.can_handle("https://example.com/file.tif")

    def test_rejects_ftp(self):
        assert not self.resolver.can_handle("ftp://example.com/file.tif")


class TestS3ResolverPassthrough(unittest.TestCase):
    def test_get_download_url(self):
        r = S3Resolver()
        url = "s3://bucket/file.tif"
        assert r.get_download_url(url) == url

    def test_get_headers_empty(self):
        assert S3Resolver().get_headers("s3://b/k") == {}


class TestS3ResolverExtractHref(unittest.TestCase):
    def test_prefers_alternate_s3(self):
        asset = {
            "href": "https://example.com/file.tif",
            "alternate": {"s3": {"href": "s3://bucket/file.tif"}},
        }
        assert S3Resolver._extract_s3_href(asset) == "s3://bucket/file.tif"

    def test_falls_back_to_href(self):
        asset = {"href": "https://example.com/file.tif"}
        assert S3Resolver._extract_s3_href(asset) == "https://example.com/file.tif"

    def test_empty_dict(self):
        assert S3Resolver._extract_s3_href({}) == ""


class TestS3ResolverDownload(unittest.TestCase):
    def test_https_returns_none(self):
        r = S3Resolver()
        assert r.download("https://bucket.s3.amazonaws.com/file.tif", Path("/tmp")) is None

    @patch("earthdaily.platform._download_resolvers.makedirs")
    def test_download_s3_quiet(self, mock_makedirs):
        mock_client = MagicMock()
        r = S3Resolver(s3_client=mock_client)
        dest = r.download("s3://mybucket/path/file.tif", Path("/output"), quiet=True)

        mock_client.download_file.assert_called_once_with(
            "mybucket",
            "path/file.tif",
            str(Path("/output/path/file.tif")),
            ExtraArgs={"RequestPayer": "requester"},
        )
        assert dest == Path("/output/path/file.tif")

    @patch("earthdaily.platform._download_resolvers.tqdm")
    @patch("earthdaily.platform._download_resolvers.makedirs")
    def test_download_s3_with_progress(self, mock_makedirs, mock_tqdm_cls):
        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": 1024}
        mock_pbar = MagicMock()
        mock_tqdm_cls.return_value.__enter__ = MagicMock(return_value=mock_pbar)
        mock_tqdm_cls.return_value.__exit__ = MagicMock(return_value=False)

        r = S3Resolver(s3_client=mock_client)
        dest = r.download("s3://mybucket/path/file.tif", Path("/output"), quiet=False)

        mock_client.head_object.assert_called_once_with(
            Bucket="mybucket",
            Key="path/file.tif",
            RequestPayer="requester",
        )
        mock_client.download_file.assert_called_once()
        assert dest == Path("/output/path/file.tif")

    @patch("earthdaily.platform._download_resolvers.makedirs")
    def test_download_no_requester_pays(self, mock_makedirs):
        mock_client = MagicMock()
        r = S3Resolver(s3_client=mock_client, requester_pays=False)
        r.download("s3://mybucket/path/file.tif", Path("/output"), quiet=True)

        mock_client.download_file.assert_called_once_with(
            "mybucket",
            "path/file.tif",
            str(Path("/output/path/file.tif")),
            ExtraArgs=None,
        )

    def test_lazy_boto3_import(self):
        r = S3Resolver()
        with patch("earthdaily.platform._download_resolvers.makedirs"):
            with self.assertRaises(ImportError):
                with patch.dict("sys.modules", {"boto3": None}):
                    r._s3_client = None
                    r._get_s3_client()

    @patch("earthdaily.platform._download_resolvers.makedirs")
    def test_file_local_path_overrides_destination(self, mock_makedirs):
        mock_client = MagicMock()
        r = S3Resolver(s3_client=mock_client)
        asset_metadata = {
            "href": "s3://mybucket/account/collection/2024/01/01/uuid/sub_folder/file.tif",
            "file:local_path": "sub_folder/file.tif",
        }
        dest = r.download(
            "s3://mybucket/account/collection/2024/01/01/uuid/sub_folder/file.tif",
            Path("/output"),
            quiet=True,
            asset_metadata=asset_metadata,
        )

        mock_client.download_file.assert_called_once_with(
            "mybucket",
            "account/collection/2024/01/01/uuid/sub_folder/file.tif",
            str(Path("/output/sub_folder/file.tif")),
            ExtraArgs={"RequestPayer": "requester"},
        )
        assert dest == Path("/output/sub_folder/file.tif")

    @patch("earthdaily.platform._download_resolvers.makedirs")
    def test_file_local_path_flat(self, mock_makedirs):
        mock_client = MagicMock()
        r = S3Resolver(s3_client=mock_client)
        asset_metadata = {"file:local_path": "result.tif"}
        dest = r.download(
            "s3://mybucket/deep/nested/path/result.tif",
            Path("/output"),
            quiet=True,
            asset_metadata=asset_metadata,
        )

        assert dest == Path("/output/result.tif")

    @patch("earthdaily.platform._download_resolvers.makedirs")
    def test_no_asset_metadata_uses_full_key(self, mock_makedirs):
        mock_client = MagicMock()
        r = S3Resolver(s3_client=mock_client)
        dest = r.download(
            "s3://mybucket/tiles/57/J/WN/2024/6/30/0/B01.jp2",
            Path("/output"),
            quiet=True,
        )

        assert dest == Path("/output/tiles/57/J/WN/2024/6/30/0/B01.jp2")

    @patch("earthdaily.platform._download_resolvers.makedirs")
    def test_empty_asset_metadata_uses_full_key(self, mock_makedirs):
        mock_client = MagicMock()
        r = S3Resolver(s3_client=mock_client)
        dest = r.download(
            "s3://mybucket/tiles/57/J/WN/2024/6/30/0/B01.jp2",
            Path("/output"),
            quiet=True,
            asset_metadata={},
        )

        assert dest == Path("/output/tiles/57/J/WN/2024/6/30/0/B01.jp2")


class TestS3ResolverLazyClient(unittest.TestCase):
    def test_creates_client_lazily(self):
        r = S3Resolver()
        assert r._s3_client is None
        mock_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            with patch("earthdaily.platform._download_resolvers.makedirs"):
                r._get_s3_client()
                mock_boto3.client.assert_called_once_with("s3")

    def test_reuses_existing_client(self):
        existing = MagicMock()
        r = S3Resolver(s3_client=existing)
        assert r._get_s3_client() is existing


class TestS3HttpsPattern(unittest.TestCase):
    def test_matches_standard_s3(self):
        assert _S3_HTTPS_PATTERN.match("my-bucket.s3.us-east-1.amazonaws.com")

    def test_matches_old_style(self):
        assert _S3_HTTPS_PATTERN.match("my-bucket.s3-us-east-1.amazonaws.com")

    def test_no_match_other(self):
        assert not _S3_HTTPS_PATTERN.match("example.com")


if __name__ == "__main__":
    unittest.main()
