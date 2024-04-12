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
from .sub.column_spec import FIColumnUnits


class FIStdCustomInfo(FIBaseCustomInfo):

    column_units: FIColumnUnits
