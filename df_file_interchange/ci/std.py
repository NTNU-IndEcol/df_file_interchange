"""
Standard custom info

Includes

* Column unit+descriptions
* Some basic optional info

"""

from pprint import pprint
from typing import Any, Literal, TypeAlias, Union
from loguru import logger

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_serializer,
    model_validator,
)


from ..file.rw import FIBaseCustomInfo
from .base import FIBaseColumnSpec, FIBaseOptionalInfo
from .unit.base import FIBaseUnit
from .unit.currency import FICurrencyUnit


class FIStdColumnSpec(FIBaseColumnSpec):
    """The standard specifications available for a single column.

    Obviously, you can extend this or extend from the base class.

    Attributes
    ----------
    unit : FIBaseUnit
        Unit specification for this column, e.g. a currency.
    """

    unit: FIBaseUnit


class FIStdOptionalInfo(FIBaseOptionalInfo):

    source: str | None = None
    version: str | int | None = None
    author: str | None = None


class FIStdCustomInfo(FIBaseCustomInfo):

    optional: FIStdOptionalInfo = FIStdOptionalInfo()
    cols: dict[Any, FIStdColumnSpec] = {}
