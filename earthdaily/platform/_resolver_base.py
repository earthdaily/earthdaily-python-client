import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Pattern
from urllib.parse import urlparse


class AssetResolver(ABC):
    """Base class for asset resolvers that handle specific domains or protocols."""

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Check if this resolver can handle the given URL."""
        pass

    @abstractmethod
    def get_download_url(self, url: str) -> str:
        """
        Transform the asset URL if needed before download.

        Returns:
            str: The URL to use for downloading
        """
        pass

    @abstractmethod
    def get_headers(self, url: str) -> dict[str, str]:
        """
        Get headers needed for downloading from this service.

        Returns:
            dict[str, str]: Headers to include in the download request
        """
        pass

    def download(
        self, url: str, output_dir: Path, quiet: bool = False, asset_metadata: dict | None = None
    ) -> Optional[Path]:
        """
        Perform a direct (non-HTTP) download for protocols like S3.

        Resolvers that support non-HTTP downloads (e.g. S3) should override this method.
        Returning None signals that the caller should fall back to the standard HTTP
        download path using get_download_url() and get_headers().

        Args:
            url: The asset URL to download.
            output_dir: Directory to save the downloaded file.
            quiet: If True, suppress progress output.
            asset_metadata: Optional STAC asset dictionary. Resolvers may use
                extra fields (e.g. ``file:local_path``) to determine the
                destination path.

        Returns:
            Path to the downloaded file, or None to fall back to HTTP download.
        """
        return None


class DefaultResolver(AssetResolver):
    """Default resolver that handles standard HTTP/HTTPS URLs with no special requirements."""

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in ["http", "https"]

    def get_download_url(self, url: str) -> str:
        return url

    def get_headers(self, url: str) -> dict[str, str]:
        return {}


class DomainPatternResolver(AssetResolver):
    """Base class for resolvers that use domain pattern matching."""

    domain_pattern: Optional[Pattern] = None

    def can_handle(self, url: str) -> bool:
        if not self.domain_pattern:
            return False

        parsed = urlparse(url)
        return bool(self.domain_pattern.match(parsed.netloc))


class EarthDailyAPIResolver(DomainPatternResolver):
    """Resolver for EarthDaily API domains."""

    domain_pattern = re.compile(r".*\.earthdaily\.com$")

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key

    def get_download_url(self, url: str) -> str:
        return url

    def get_headers(self, url: str) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
