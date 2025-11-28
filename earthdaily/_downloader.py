import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout
from tqdm import tqdm

from earthdaily._eds_logging import LoggerConfig
from earthdaily.exceptions import DownloadValidationError, UnsupportedAssetException

logger = LoggerConfig(logger_name=__name__).get_logger()


class HttpDownloader:
    """
    A generic file downloader for HTTP GET requests.

    This class allows downloading files from any domain and file extension,
    provided the protocol (e.g., HTTP, HTTPS) is supported.

    Attributes:
        timeout (int): Timeout for download requests, in seconds.
        supported_protocols (List[str]): List of supported protocols (e.g., ["http", "https"]).
        allow_redirects (bool): Whether to allow URL redirects during file downloads.
    """

    timeout = 120  # Seconds
    MIN_CHUNK_SIZE = 1
    MAX_CHUNK_SIZE = 100 * 1024 * 1024  # 100 MB

    def __init__(
        self,
        supported_protocols: List[str],
        allow_redirects: bool = True,
    ) -> None:
        """
        Initialize the FileDownloader.

        Args:
            supported_protocols: List of protocols allowed for downloading.
                Example: ["http", "https"].
            allow_redirects: Whether to allow URL redirects during download.
        """
        self.supported_protocols = supported_protocols
        self.allow_redirects = allow_redirects

    def _validate_chunk_size(self, chunk_size: int) -> None:
        if not isinstance(chunk_size, int):
            raise DownloadValidationError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")
        if chunk_size < self.MIN_CHUNK_SIZE:
            raise DownloadValidationError(
                f"chunk_size must be at least {self.MIN_CHUNK_SIZE} byte(s), got {chunk_size}"
            )
        if chunk_size > self.MAX_CHUNK_SIZE:
            raise DownloadValidationError(
                f"chunk_size must not exceed {self.MAX_CHUNK_SIZE} bytes (100 MB), got {chunk_size}"
            )

    def _validate_total_size(self, expected_size: int, actual_size: int, file_url: str) -> None:
        if expected_size > 0 and actual_size != expected_size:
            raise DownloadValidationError(
                f"Downloaded file size mismatch for {file_url}: expected {expected_size} bytes, got {actual_size} bytes"
            )

    def is_supported_file(self, file_url: str) -> bool:
        """
        Check if the file URL has a supported protocol for download.

        Args:
            file_url: URL of the file to check.

        Returns:
            bool: True if the file's protocol is supported, False otherwise.
        """
        parsed_url = urlparse(file_url, allow_fragments=False)
        return parsed_url.scheme in self.supported_protocols

    def download_file(
        self,
        file_url: str,
        save_location: Path,
        chunk_size: int = 8192,
        quiet: bool = False,
        continue_on_error: bool = False,
    ) -> Optional[Tuple[str, Path]]:
        """
        Download a file from a URL and save it to the specified location.

        Args:
            file_url: URL of the file to download.
            save_location: Path to save the downloaded file.
            chunk_size: Size of chunks to use for streaming the download (default: 8192 bytes).
            quiet: Whether to suppress the progress bar during the download (default: False).
            continue_on_error: If True, continue downloading other files if an error occurs.
                               If False, raise the error (default: False).

        Returns:
            Tuple containing the file URL and the path to the saved file, or None if download fails.

        Raises:
            UnsupportedAssetException: If the file is not supported for download.
            HTTPError: If the HTTP request returned an unsuccessful status code.
            RequestException: For general request-related errors.
            OSError: For file system-related issues (e.g., issues saving the file).
        """
        if not self.is_supported_file(file_url):
            raise UnsupportedAssetException(f"FileDownloader does not support the file at {file_url}")

        headers = self.get_request_headers()
        parsed_url = urlparse(file_url, allow_fragments=False)
        file_name = os.path.basename(parsed_url.path)

        try:
            self._validate_chunk_size(chunk_size)
            with requests.get(
                file_url,
                stream=True,
                allow_redirects=self.allow_redirects,
                headers=headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()

                if self.allow_redirects and response.url != file_url:
                    final_url = urlparse(response.url)
                    final_filename = os.path.basename(final_url.path)
                    if final_filename and "." in final_filename:
                        file_name = final_filename

                file_path = Path(save_location).joinpath(file_name)
                total_size = int(response.headers.get("content-length", 0))

                downloaded_size = 0
                with open(file_path, "wb") as file:
                    if not quiet:
                        with tqdm(
                            desc=file_name,
                            total=total_size,
                            unit="iB",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as progress_bar:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                size = file.write(chunk)
                                downloaded_size += size
                                progress_bar.update(size)
                    else:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            downloaded_size += file.write(chunk)

                self._validate_total_size(total_size, downloaded_size, file_url)

                return file_url, file_path

        except (HTTPError, ConnectionError, Timeout, RequestException, OSError, DownloadValidationError) as err:
            error_message = f"Error downloading {file_url}: {err}"
            if continue_on_error:
                logger.warning(error_message)
                return None
            else:
                raise err

    def get_request_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for the download request.

        Returns:
            Dict[str, str]: A dictionary containing any necessary headers for the request.
        """
        return {}
