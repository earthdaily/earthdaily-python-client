import os
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from earthdaily._eds_logging import LoggerConfig
from earthdaily.exceptions import UnsupportedAssetException

logger = LoggerConfig(logger_name=__name__).get_logger()


class HttpUploader:
    """
    A configurable HTTP uploader class that supports file uploads with progress tracking.

    This class provides methods for uploading files to specified URLs, with options
    for progress tracking, error handling, and protocol restrictions.

    Attributes:
        supported_protocols (List[str]): List of supported URL protocols.
        timeout (int): Timeout for upload requests in seconds.
        chunk_size (int): Size of chunks for file reading and uploading.
        quiet (bool): If True, suppresses progress bar display.
        continue_on_error (bool): If True, continues uploading on non-fatal errors.
    """

    def __init__(
        self,
        supported_protocols: List[str] = ["http", "https"],
        timeout: int = 120,
        chunk_size: int = 8192,
        quiet: bool = False,
        continue_on_error: bool = False,
    ) -> None:
        """
        Initialize the HttpUploader with the given configuration.

        Args:
            supported_protocols (List[str]): List of supported URL protocols.
            timeout (int): Timeout for upload requests in seconds.
            chunk_size (int): Size of chunks for file reading and uploading.
            quiet (bool): If True, suppresses progress bar display.
            continue_on_error (bool): If True, continues uploading on non-fatal errors.
        """
        self.supported_protocols = supported_protocols
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.quiet = quiet
        self.continue_on_error = continue_on_error

    def _check_protocol(self, url: str) -> None:
        """
        Check if the given URL uses a supported protocol.

        Args:
            url (str): The URL to check.

        Raises:
            UnsupportedAssetException: If the URL protocol is not supported.
        """
        parsed_url = urlparse(url)
        if parsed_url.scheme not in self.supported_protocols:
            raise UnsupportedAssetException(f"Unsupported protocol: {parsed_url.scheme}")

    def upload_file(self, file_path: Path, upload_url: str) -> bool:
        """
        Upload a file to the specified URL with progress tracking.

        Args:
            file_path (str): Path to the file to be uploaded.
            upload_url (str): URL to upload the file to.

        Returns:
            bool: True if upload was successful, False otherwise.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            RequestException: For any network-related errors during upload when continue_on_error is False.
        """
        self._check_protocol(upload_url)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)

        with open(file_path, "rb") as file:
            data = file.read()

            with tqdm(total=file_size, unit="B", unit_scale=True, disable=self.quiet, desc="Uploading") as progress_bar:
                try:
                    # Non-chunked upload
                    response = requests.put(
                        upload_url,
                        data=data,
                        timeout=self.timeout,
                    )
                    progress_bar.update(file_size)

                    response.raise_for_status()
                    return True
                except RequestException as e:
                    logger.warning(f"Error during upload: {str(e)}")
                    if not self.continue_on_error:
                        raise
                    return False
