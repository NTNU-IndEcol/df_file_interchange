"""Base class for doing extra validated information

Any "extra info" classes must derive from this.
"""

from typing import Any
from loguru import logger  # noqa: F401

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    model_validator,
)


class FIBaseExtraInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_classname(cls) -> str:
        return cls.__name__

    @computed_field
    @property
    def classname(self) -> str:
        """Ensures classname is included in serialization

        Returns
        -------
        str
            Our classname
        """

        return self.get_classname()

    @model_validator(mode="before")
    @classmethod
    def model_validator_extra_info(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "classname" in data.keys():
                del data["classname"]
        return data
