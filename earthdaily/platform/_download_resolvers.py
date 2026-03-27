from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from os import makedirs
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from earthdaily.platform._resolver_base import AssetResolver, DomainPatternResolver

logger = logging.getLogger(__name__)

_EARTHDATA_DOMAINS = [
    "data.lpdaac.earthdatacloud.nasa.gov",
    "data.laadsdaac.earthdatacloud.nasa.gov",
    "cmr.earthdata.nasa.gov",
    "ladsweb.modaps.eosdis.nasa.gov",
    "data.nsidc.earthdatacloud.nasa.gov",
]

_S3_HTTPS_PATTERN = re.compile(r"(s3-|s3\.)?(.*)\.amazonaws\.com")


class EarthDataResolver(DomainPatternResolver):
    """Resolver for NASA EarthData assets.

    Accepts a JWT token directly.  The token is validated for expiry before
    each request; callers are responsible for providing a non-expired token.
    """

    needs_credentials = True

    domain_pattern = re.compile("|".join(re.escape(d) for d in _EARTHDATA_DOMAINS))

    def __init__(self, token: str) -> None:
        self._raw_token = token

    @classmethod
    def from_config(cls, config: Any) -> EarthDataResolver | None:
        token = getattr(config, "earthdata_token", "") or ""
        return cls(token=token) if token else None

    @staticmethod
    def _add_base64_padding(data: str) -> str:
        return data + "=" * (-len(data) % 4)

    @classmethod
    def _validate_jwt(cls, jwt_token: str) -> dict:
        """Decode and validate a JWT token, raising on expiry or bad format."""
        parts = jwt_token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid EarthData JWT: expected three dot-separated parts.")

        try:
            decoded = base64.urlsafe_b64decode(cls._add_base64_padding(parts[1])).decode("utf-8")
        except Exception as exc:
            raise ValueError("Invalid EarthData JWT: payload is not valid Base64.") from exc

        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid EarthData JWT: payload is not valid JSON.") from exc

        exp = payload.get("exp")
        if not exp:
            raise ValueError("Invalid EarthData JWT: missing 'exp' claim.")
        if time.time() > exp:
            raise ValueError(f"EarthData JWT expired on {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(exp))}.")
        return payload

    def get_download_url(self, url: str) -> str:
        return url

    def get_headers(self, url: str) -> dict[str, str]:
        self._validate_jwt(self._raw_token)
        return {"Authorization": f"Bearer {self._raw_token}"}


class EUMETSATResolver(DomainPatternResolver):
    """Resolver for EUMETSAT assets.

    Performs OAuth2 ``client_credentials`` token exchange against
    ``https://api.eumetsat.int/token`` and caches the bearer token with a
    60-second safety buffer before expiry.
    """

    needs_credentials = True

    domain_pattern = re.compile(r"api\.eumetsat\.int$")

    _TOKEN_URL = "https://api.eumetsat.int/token"

    def __init__(self, client_id: str, client_secret: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._bearer_token: str | None = None
        self._token_expires: datetime | None = None

    @classmethod
    def from_config(cls, config: Any) -> EUMETSATResolver | None:
        client_id = getattr(config, "eumetsat_client_id", "") or ""
        client_secret = getattr(config, "eumetsat_client_secret", "") or ""
        if client_id and client_secret:
            return cls(client_id=client_id, client_secret=client_secret)
        return None

    def _refresh_token_if_needed(self) -> None:
        if self._bearer_token and self._token_expires and self._token_expires > datetime.now(timezone.utc):
            return

        response = requests.post(
            self._TOKEN_URL,
            auth=HTTPBasicAuth(self._client_id, self._client_secret),
            timeout=15,
            data={"grant_type": "client_credentials"},
        )
        response.raise_for_status()
        body = response.json()
        self._bearer_token = body["access_token"]
        self._token_expires = datetime.now(timezone.utc) + timedelta(seconds=int(body["expires_in"]) - 60)

    def get_download_url(self, url: str) -> str:
        return url

    def get_headers(self, url: str) -> dict[str, str]:
        self._refresh_token_if_needed()
        return {"Authorization": f"Bearer {self._bearer_token}"}


class S3Resolver(AssetResolver):
    """Resolver for S3 assets (``s3://`` and S3 HTTPS URLs).

    Uses a ``boto3`` S3 client for ``s3://`` URLs and falls back to plain HTTP
    for S3 HTTPS URLs.

    When constructed without an explicit *s3_client*, a default client is
    lazily created from the environment's AWS credential chain on first
    download.  This requires ``boto3`` to be installed
    (``pip install 'earthdaily[download]'``).

    Parameters
    ----------
    s3_client :
        A ``boto3`` S3 client (``boto3.client("s3")``).  If ``None``, one is
        created lazily using default AWS credentials.
    requester_pays :
        Whether to send ``RequestPayer: requester`` for ``s3://`` downloads.
    """

    needs_credentials = False

    def __init__(
        self,
        s3_client: Any = None,
        requester_pays: bool = True,
    ) -> None:
        self._s3_client = s3_client
        self._requester_pays = requester_pays

    def _get_s3_client(self) -> Any:
        """Return the S3 client, lazily creating a default one if needed."""
        if self._s3_client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 downloads. Install it with: pip install 'earthdaily[download]'"
                )
            self._s3_client = boto3.client("s3")
        return self._s3_client

    @staticmethod
    def _extract_s3_href(asset_dict: dict) -> str:
        """Prefer ``alternate.s3.href`` over the top-level ``href``."""
        alt = asset_dict.get("alternate", {}).get("s3", {}).get("href", "")
        return alt if alt else asset_dict.get("href", "")

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url, allow_fragments=False)
        if parsed.scheme == "s3":
            return not parsed.path.endswith("/")
        if parsed.scheme == "https" and _S3_HTTPS_PATTERN.match(parsed.netloc):
            return not parsed.path.endswith("/")
        return False

    def get_download_url(self, url: str) -> str:
        return url

    def get_headers(self, url: str) -> dict[str, str]:
        return {}

    def download(
        self, url: str, output_dir: Path, quiet: bool = False, asset_metadata: dict | None = None
    ) -> Path | None:
        """Download an S3 asset directly via boto3.

        For ``s3://`` URLs this uses ``s3_client.download_file``.  For HTTPS
        S3 URLs the method returns ``None`` to fall back to the standard HTTP
        download path.

        When *asset_metadata* contains a ``file:local_path`` field the file is
        saved at ``output_dir / <file:local_path>`` instead of the default
        flat filename derived from the S3 key.
        """
        parsed = urlparse(url, allow_fragments=False)

        if parsed.scheme != "s3":
            return None

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        local_path = (asset_metadata or {}).get("file:local_path")
        if local_path:
            dest = output_dir / local_path
        else:
            dest = output_dir / key

        makedirs(dest.parent, exist_ok=True)

        extra_args: dict | None = None
        if self._requester_pays:
            extra_args = {"RequestPayer": "requester"}

        s3 = self._get_s3_client()
        logger.info("Downloading %s to %s", url, dest)

        if quiet:
            s3.download_file(bucket, key, str(dest), ExtraArgs=extra_args)
        else:
            head = s3.head_object(Bucket=bucket, Key=key, **(extra_args or {}))
            total = head["ContentLength"]
            with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                s3.download_file(bucket, key, str(dest), ExtraArgs=extra_args, Callback=pbar.update)

        return dest
