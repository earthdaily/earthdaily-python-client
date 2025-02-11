import json
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class BaseModel:
    """
    Base model class for Earth Data Store (EDS) models.

    This class provides common functionality for all EDS models,
    including methods to convert the model to a dictionary or JSON string.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model instance to a dictionary.

        Returns:
        -------
        Dict[str, Any]:
            A dictionary representation of the model, excluding None values.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        """
        Convert the model instance to a JSON string.

        Returns:
        -------
        str:
            A JSON string representation of the model, excluding None values.
        """
        return json.dumps(self.to_dict())


@dataclass
class BaseRequest(BaseModel):
    """
    Base class for all request models in the Earth Data Store (EDS) library.

    This class inherits from BaseModel and should be used as a parent class
    for all specific request models.
    """

    pass


@dataclass
class BaseResponse(BaseModel):
    """
    Base class for all response models in the Earth Data Store (EDS) library.

    This class inherits from BaseModel and should be used as a parent class
    for all specific response models.
    """

    pass
