"""
Structured custom info

Includes

* Column unit+descriptions
* Some basic optional info

"""

from pprint import pprint
from typing import Any, Literal, TypeAlias, Union, Self, Iterator
from loguru import logger

# from contextlib import contextmanager
# from contextvars import ContextVar
from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_serializer,
    model_serializer,
    model_validator,
    field_validator,
    ValidationInfo,
    SerializeAsAny,
)




class FIBaseCustomInfo(BaseModel):
    """Wrapper class to store user custom info

    N.B. This, and any descendent, MUST be able to deserialise based on a
    provided dictionary!

    A descendent of this is usually supplied as an object when writing a file to
    include additional metadata. When reading, a class is passed as a parameter
    and an object will be instantiated upon reading.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unstructured_data: dict = {}

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