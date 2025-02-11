class EDSAPIError(Exception):
    """
    Exception raised when an API request to the EDS platform fails.

    Attributes:
        message (str): Explanation of the error.
        status_code (int, optional): HTTP status code of the failed API response.
        body (str, optional): Body of the failed API response, often containing additional error details.

    Args:
        message (str): Explanation of the error.
        status_code (int, optional): HTTP status code of the failed API response.
        body (str, optional): Body of the failed API response, often containing additional error details.
    """

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class UnsupportedAssetException(Exception):
    """Exception raised when an asset is not supported for download."""

    pass
