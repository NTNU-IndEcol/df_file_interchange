"""Standard extra info"""

from typing import Any
from loguru import logger

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    model_validator,
    field_validator,
    ValidationInfo,
    SerializeAsAny,
)

from .base import FIBaseExtraInfo


class FIStdExtraInfo(FIBaseExtraInfo):
    """Standard extra info for a dataframe

    Attributes
    ----------
    author : str | None
        Default None.
    source : str | None
        Where it came from. Default None.
    """

    author: str | None = None
    source: str | None = None
