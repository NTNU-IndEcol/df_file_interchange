"""
Standard custom info

Includes

* Column unit+descriptions

"""

from pprint import pprint
from typing import Any, Literal, TypeAlias, Union

import numpy as np
import pandas as pd
from loguru import logger

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_serializer,
    model_validator,
)


from ..file.rw import FIBaseCustomInfo
from .sub.column_spec import FIBaseUnit, FICurrencyUnit



class FIBaseColumnSpec(BaseModel):

    pass


class FIStdColumnSpec(FIBaseColumnSpec):
    """The standard specifications available for a single column.

    Obviously, you can extend this or extend from the base class.

    These are collected together to provide the custom metadata for all columns
    using FIStdColumnSetSpec.

    Attributes
    ----------
    unit : FIBaseUnit
        Unit specification for this column.
    """

    unit: FIBaseUnit



class FIStdColumnSetSpec(FIBaseCustomInfo):

    cols: dict[Any, FIBaseColumnSpec] = {}



