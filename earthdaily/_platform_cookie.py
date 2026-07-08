import atexit
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from http.cookies import SimpleCookie
from pathlib import Path

from earthdaily._api_requester import APIRequester
from earthdaily._eds_config import AssetAccessMode, EDSConfig
from earthdaily._eds_logging import LoggerConfig
from earthdaily._http_client import HTTPRequest
from earthdaily.exceptions import EDSAPIError

logger = LoggerConfig(logger_name=__name__).get_logger()

COOKIE_NAME = "PLATFORM_ASSET"
COOKIE_DOMAIN = ".earthdaily.com"
COOKIE_PATH = "/platform/v1/asset"
NETSCAPE_HEADER = "# Netscape HTTP Cookie File\n"
DEFAULT_TTL_SECONDS = 3600
DEFAULT_REFRESH_MARGIN_SECONDS = 300
DEFAULT_REFRESH_RETRY_SECONDS = 60


@dataclass
class PlatformAssetToken:
    """A platform asset cookie value and its expiry.

    Attributes
    ----------
    value : str
        The raw cookie token value.
    expires_at : float
        Unix timestamp (seconds) when the cookie expires.
    netscape_line : str
        The pre-built Netscape cookie file line derived from the server response.
    """

    value: str
    expires_at: float
    netscape_line: str = ""


class PlatformAssetCookieClient:
    """HTTP wrapper around `POST /platform/v1/asset/cookie`.

    The endpoint returns HTTP 204 with no body. The cookie is delivered
    entirely via the ``Set-Cookie`` response header.

    Example response:
        HTTP/2 204
        set-cookie: PLATFORM_ASSET=eyJhbGci...<jwt>...;
                    Domain=.earthdaily.com;
                    HttpOnly;
                    Max-Age=3600;
                    Path=/platform/v1/asset;
                    SameSite=strict;
                    Secure

    Parameters
    ----------
    api_requester : APIRequester
        The API requester used to send authenticated requests.
    """

    def __init__(self, api_requester: APIRequester):
        self.api_requester = api_requester

    def fetch(self, ttl_seconds: int) -> PlatformAssetToken:
        """Fetch a new platform asset cookie.

        Parameters
        ----------
        ttl_seconds : int
            Desired cookie lifetime, sent as the `max_age` field of the request
            body. The server may honor or clamp this value.

        Returns
        -------
        PlatformAssetToken
            The fetched token and its expiry.

        Raises
        ------
        EDSAPIError
            If the cookie endpoint returns a non-success status or the response
            is missing the expected `PLATFORM_ASSET` cookie.
        """
        url = f"{self.api_requester.base_url}/platform/v1/asset/cookie"
        request = HTTPRequest(method="POST", url=url, body={"max_age": ttl_seconds})
        request_time = time.time()
        response = self.api_requester.send_request(request)

        if not (200 <= response.status_code < 300):
            raise EDSAPIError(
                f"Platform asset cookie request failed with status {response.status_code}: {response.body}",
                status_code=response.status_code,
                body=response.body,
            )

        set_cookie = response.headers.get("Set-Cookie") if response.headers else None
        if not set_cookie:
            raise EDSAPIError(
                "Platform asset cookie response is missing the 'Set-Cookie' header",
                status_code=response.status_code,
                body=response.body,
            )

        return self._parse_set_cookie(set_cookie, ttl_seconds=ttl_seconds, request_time=request_time)

    @staticmethod
    def _parse_set_cookie(set_cookie: str, ttl_seconds: int, request_time: float) -> PlatformAssetToken:
        """Parse a `Set-Cookie` header looking for the `PLATFORM_ASSET` cookie."""
        jar = SimpleCookie()
        jar.load(set_cookie)

        morsel = jar.get(COOKIE_NAME)
        if morsel is None:
            raise EDSAPIError(
                f"Platform asset cookie response does not contain a '{COOKIE_NAME}' cookie",
            )

        max_age = morsel["max-age"]
        if max_age:
            try:
                expires_at = request_time + int(max_age)
            except ValueError:
                expires_at = request_time + ttl_seconds
        else:
            expires_at = request_time + ttl_seconds

        domain = morsel["domain"] or COOKIE_DOMAIN
        path = morsel["path"] or COOKIE_PATH
        secure = "TRUE" if morsel["secure"] else "FALSE"
        netscape_line = f"{domain}\tTRUE\t{path}\t{secure}\t{int(expires_at)}\t{morsel.key}\t{morsel.value}\n"

        return PlatformAssetToken(value=morsel.value, expires_at=expires_at, netscape_line=netscape_line)


class PlatformCookieManager:
    """Writes and refreshes the platform asset cookie file.

    The manager owns a single in-memory token plus an optional background
    daemon thread that refreshes the token before it expires. The cookie file
    location is decided by the caller and passed into `get_cookie_file()`.

    Parameters
    ----------
    api_requester : APIRequester
        The API requester used by the cookie client.
    config : EDSConfig
        The client configuration, checked for feature enablement and
        `asset_access_mode`.
    """

    def __init__(self, api_requester: APIRequester, config: EDSConfig):
        self.api_requester = api_requester
        self.config = config
        self._cookie_client = PlatformAssetCookieClient(api_requester)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._token: PlatformAssetToken | None = None
        self._path: Path | None = None
        self._ttl_seconds: int = DEFAULT_TTL_SECONDS
        self._atexit_registered = False

    def get_cookie_file(
        self,
        path: str | Path,
        force_refresh: bool = False,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> Path:
        """Return a path to a valid Netscape cookie file, refreshing if needed.

        Parameters
        ----------
        path : str | Path
            Path to the cookie file. Required; no default is provided so users
            consciously pick a location. The parent directory must already
            exist.
        force_refresh : bool, default False
            If True, force a fresh cookie fetch even if the current token is
            still valid.
        ttl_seconds : int, default 3600
            Desired cookie lifetime passed to the cookie endpoint as `max_age`.

        Returns
        -------
        Path
            The resolved cookie file path.

        Raises
        ------
        ValueError
            If `path` is missing or its parent directory does not exist, or if
            the cookie feature is not enabled or the client is not in
            proxy-urls mode.
        """
        self._validate_enabled()

        if path is None:
            raise ValueError("path is required; pick a location such as '/tmp/earthdaily-platform-cookies.txt'")

        resolved = Path(path).expanduser().resolve()
        parent = resolved.parent
        if not parent.exists():
            raise ValueError(
                f"Parent directory does not exist: {parent}. Create it first; "
                "get_platform_cookie_file() does not create directories."
            )
        if not parent.is_dir():
            raise ValueError(f"Parent path is not a directory: {parent}")

        with self._lock:
            self._path = resolved
            self._ttl_seconds = ttl_seconds

            if force_refresh or self._needs_refresh_locked():
                self._refresh_locked(ttl_seconds=ttl_seconds)

            self._register_atexit_locked()

            if self.config.platform_cookie_auto_refresh:
                self._start_auto_refresh_locked()

            return resolved

    def get_cookie_value(self) -> str:
        """Return the current cookie token value, refreshing if needed.

        Raises
        ------
        ValueError
            If the cookie feature is not enabled or the client is not in
            proxy-urls mode.
        """
        self._validate_enabled()

        with self._lock:
            if self._token is None or self._needs_refresh_locked():
                self._refresh_locked(ttl_seconds=self._ttl_seconds)
            assert self._token is not None
            return self._token.value

    def start_auto_refresh(self) -> None:
        """Start the background refresh thread if it is not already running."""
        self._validate_enabled()
        with self._lock:
            if self._path is None:
                raise ValueError("Call get_platform_cookie_file(path=...) before starting the auto-refresh loop")
            self._start_auto_refresh_locked()

    def stop_auto_refresh(self) -> None:
        """Stop the background refresh thread if it is running."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        with self._lock:
            self._thread = None
            self._stop_event.clear()

    def close(self) -> None:
        """Stop the background refresh thread and release internal state."""
        self.stop_auto_refresh()

        with self._lock:
            self._path = None
            self._token = None

    def _validate_enabled(self) -> None:
        if not self.config.platform_cookie_enabled:
            raise ValueError(
                "Platform cookie support is not enabled. "
                "Set EDSConfig(platform_cookie_enabled=True) to use get_platform_cookie_file()."
            )
        if self.config.asset_access_mode != AssetAccessMode.PROXY_URLS:
            raise ValueError(
                "Platform cookie file only applies when asset_access_mode='proxy-urls'. "
                f"Current mode: {self.config.asset_access_mode}."
            )

    def _needs_refresh_locked(self) -> bool:
        if self._token is None or self._path is None or not self._path.exists():
            return True
        return time.time() >= (self._token.expires_at - DEFAULT_REFRESH_MARGIN_SECONDS)

    def _refresh_locked(self, ttl_seconds: int) -> None:
        assert self._path is not None
        token = self._cookie_client.fetch(ttl_seconds=ttl_seconds)
        self._write_cookie_file(self._path, token)
        self._token = token

    @staticmethod
    def _write_cookie_file(path: Path, token: PlatformAssetToken) -> None:
        """Atomically upsert the cookie into a Netscape cookie file.

        Preserves any existing cookies in the file, replacing only the entry
        for this cookie (matched by name) and adding the Netscape header if absent.
        """
        parent = path.parent
        cookie_name = token.netscape_line.rstrip("\n").split("\t")[5]

        if path.exists():
            existing_lines = path.read_text().splitlines(keepends=True)
            has_header = False
            kept = []
            for line in existing_lines:
                if line.startswith("#"):
                    if NETSCAPE_HEADER.strip() in line:
                        has_header = True
                    kept.append(line)
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 7 and parts[5] == cookie_name:
                    continue
                kept.append(line)
        else:
            kept = []
            has_header = False

        if not has_header:
            kept.insert(0, NETSCAPE_HEADER)

        contents = "".join(kept) + token.netscape_line

        fd, tmp_name = tempfile.mkstemp(prefix=".platform-cookies-", dir=str(parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(contents)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, path)
        except Exception:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _register_atexit_locked(self) -> None:
        if self._atexit_registered:
            return
        atexit.register(self._atexit_cleanup)
        self._atexit_registered = True

    def _atexit_cleanup(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _start_auto_refresh_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop,
            daemon=True,
            name="earthdaily-platform-cookie-refresh",
        )
        self._thread.start()

    def _refresh_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    if self._path is None:
                        return
                    if self._needs_refresh_locked():
                        self._refresh_locked(ttl_seconds=self._ttl_seconds)
                    next_expires_at = self._token.expires_at if self._token else time.time() + self._ttl_seconds

                refresh_at = next_expires_at - DEFAULT_REFRESH_MARGIN_SECONDS
                sleep_seconds = max(30.0, refresh_at - time.time())
            except Exception:
                logger.warning("Failed to refresh platform cookie; retrying later", exc_info=True)
                sleep_seconds = DEFAULT_REFRESH_RETRY_SECONDS

            if self._stop_event.wait(timeout=sleep_seconds):
                return
