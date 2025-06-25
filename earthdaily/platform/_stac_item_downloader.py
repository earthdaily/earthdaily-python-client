from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

from pystac import Item

from earthdaily._api_requester import APIRequester
from earthdaily._downloader import HttpDownloader
from earthdaily._eds_logging import LoggerConfig
from earthdaily.platform._stac_item_asset import get_resolver_for_url

logger = LoggerConfig(logger_name=__name__).get_logger()


class ItemDownloader:
    """
    Downloader for assets from PySTAC Items.

    This class provides functionality to download assets from PySTAC Items,
    handling different types of assets and domains using appropriate resolvers.
    """

    def __init__(
        self,
        max_workers: int = 5,
        timeout: int = 120,
        allow_redirects: bool = True,
        api_requester: Optional[APIRequester] = None,
    ):
        """
        Initialize the ItemDownloader.

        Args:
            max_workers: Maximum number of concurrent downloads
            timeout: Timeout for download requests, in seconds
            allow_redirects: Whether to allow URL redirects during download
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.allow_redirects = allow_redirects
        self.api_requester = api_requester

    def download_assets(
        self,
        item: Union[Item, dict],
        asset_keys: Optional[list[str]] = None,
        output_dir: Union[str, Path] = ".",
        quiet: bool = False,
        continue_on_error: bool = False,
        href_type: str = "href",
    ) -> dict[str, Path]:
        """
        Download selected assets from a PySTAC Item.

        Args:
            item: PySTAC Item or dictionary with STAC Item structure
            asset_keys: List of asset keys to download. If None, downloads all assets.
            output_dir: Directory to save downloaded files
            quiet: Whether to suppress progress bars
            continue_on_error: If True, continue downloading if an error occurs
            href_type: Specifies which href to use from asset's alternate URLs.
                       Example: "alternate.download.href" would look for that path in the asset dictionary
                       If None, uses the default asset href.

        Returns:
            Dict mapping asset keys to downloaded file paths
        """
        if isinstance(item, Item):
            item_dict = item.to_dict()
        elif isinstance(item, dict):
            try:
                # Attempt to validate the item structure
                Item.from_dict(item)
                item_dict = item
            except Exception as e:
                logger.error(f"Invalid item structure: {e}")
                raise ValueError("Provided item is not a valid PySTAC Item or dictionary representation")
        else:
            raise TypeError("Item must be a PySTAC Item or a dictionary representation of a STAC Item")

        assets_to_download = {}
        assets_dict = item_dict.get("assets", {})

        if asset_keys is None:
            # Download all assets
            assets_to_download = assets_dict
        else:
            for key in asset_keys:
                if key in assets_dict:
                    assets_to_download[key] = assets_dict[key]
                else:
                    if not continue_on_error:
                        raise ValueError(f"Asset key '{key}' not found in item")
                    logger.warning(f"Asset key '{key}' not found in item, skipping")

        if not assets_to_download:
            logger.warning("No assets found to download")
            return {}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        download_tasks = []
        for key, asset_dict in assets_to_download.items():
            href = self._get_asset_href(asset_dict, href_type)

            if not href:
                msg = f"No valid href found for asset '{key}'"
                if href_type:
                    msg += f" with href_type '{href_type}'"
                if continue_on_error:
                    logger.warning(f"{msg}, skipping")
                    continue
                else:
                    raise ValueError(msg)

            resolver = get_resolver_for_url(href, api_requester=self.api_requester)
            download_url = resolver.get_download_url(href)
            headers = resolver.get_headers(href)

            download_tasks.append((key, download_url, headers, output_path))

        results = {}

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_key = {
                    executor.submit(
                        self._download_single_asset,
                        url=url,
                        headers=headers,
                        output_path=output_path,
                        quiet=quiet,
                        continue_on_error=continue_on_error,
                    ): key
                    for key, url, headers, output_path in download_tasks
                }

                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        file_path = future.result()
                        if file_path:
                            results[key] = file_path
                    except Exception as e:
                        if not continue_on_error:
                            raise e
                        logger.warning(f"Error downloading asset '{key}': {str(e)}")
        else:
            for key, url, headers, output_path in download_tasks:
                try:
                    file_path = self._download_single_asset(
                        url=url,
                        headers=headers,
                        output_path=output_path,
                        quiet=quiet,
                        continue_on_error=continue_on_error,
                    )
                    if file_path:
                        results[key] = file_path
                except Exception as e:
                    if not continue_on_error:
                        raise e
                    logger.warning(f"Error downloading asset '{key}': {str(e)}")

        return results

    def _get_asset_href(self, asset: dict, href_type: str) -> Optional[str]:
        """
        Get the appropriate href from an asset based on href_type.

        Args:
            asset: Asset dictionary
            href_type: The type of href to retrieve. Can be a nested path like "alternate.download.href"

        Returns:
            The href string or None if not found
        """
        # If href_type is None, use default href
        if not href_type:
            return asset.get("href")

        # Handle nested path in href_type (e.g., "alternate.download.href")
        current = asset
        for part in href_type.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]

        return current if isinstance(current, str) else None

    def _download_single_asset(
        self,
        url: str,
        headers: dict[str, str],
        output_path: Path,
        quiet: bool = False,
        continue_on_error: bool = False,
    ) -> Optional[Path]:
        """
        Download a single asset.

        Args:
            url: URL of the asset to download
            headers: HTTP headers for the download request
            output_path: Directory to save the downloaded file
            quiet: Whether to suppress the progress bar
            continue_on_error: Whether to continue on error

        Returns:
            Path to the downloaded file, or None if download failed and continue_on_error is True
        """
        custom_downloader = CustomHeadersDownloader(
            supported_protocols=["http", "https"],
            allow_redirects=self.allow_redirects,
            custom_headers=headers,
        )

        try:
            result = custom_downloader.download_file(
                file_url=url,
                save_location=output_path,
                quiet=quiet,
                continue_on_error=continue_on_error,
            )

            if result:
                _, file_path = result
                return file_path
            return None

        except Exception as e:
            if continue_on_error:
                logger.warning(f"Error downloading {url}: {str(e)}")
                return None
            raise


class CustomHeadersDownloader(HttpDownloader):
    """HTTP downloader that uses custom headers for requests."""

    def __init__(
        self,
        supported_protocols: list[str],
        allow_redirects: bool = True,
        custom_headers: Optional[dict[str, str]] = None,
    ):
        super().__init__(supported_protocols, allow_redirects)
        self.custom_headers = custom_headers or {}

    def get_request_headers(self) -> dict[str, str]:
        """
        Get HTTP headers for the download request.

        Returns:
            Dict[str, str]: A dictionary containing necessary headers for the request.
        """
        headers = super().get_request_headers()
        headers.update(self.custom_headers)
        return headers
