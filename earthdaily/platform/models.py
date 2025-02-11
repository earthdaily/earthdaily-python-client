from dataclasses import dataclass


@dataclass
class UploadConfig:
    """
    Configuration for upload operations in the Earth Data Store (EDS) platform.

    Attributes:
    -----------
    type: str
        The type of upload configuration, e.g., "PRESIGNED_URL".
    presigned_url: Optional[str]
        The presigned URL for upload.
    """

    type: str
    presigned_url: str
