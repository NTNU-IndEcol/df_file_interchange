"""
Structured custom info

Includes

* Column unit+descriptions
* Some basic optional info

"""

from pprint import pprint
from typing import Any, Literal, TypeAlias, Union, Self, Iterator
from loguru import logger

from contextlib import contextmanager
from contextvars import ContextVar
from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_serializer,
    model_validator,
    field_validator,
    ValidationInfo,
)


from ..file.rw import _init_context_var, init_context, FIBaseCustomInfo

from . import unit
from .unit.base import FIBaseUnit
from .unit.currency import FICurrencyUnit



class FIBaseExtraInfo(BaseModel):
    pass


class FIStdExtraInfo(FIBaseExtraInfo):
    
    some_test_field: str = "test"





class FIStructuredCustomInfo(FIBaseCustomInfo):

    # Extra meta info that applies to the whole table
    extra_info: FIBaseExtraInfo

    # Columnwise unit info
    col_units: dict[Any, FIBaseUnit] = {}

    def __init__(self, /, **data: Any) -> None:

        # Get parent to do its bit.
        super().__init__(**data)

        self.__pydantic_validator__.validate_python(
            data,
            self_instance=self,
            context=_init_context_var.get(),
        )

    @field_validator('extra_info')
    @classmethod
    def validator_extra_info(cls, value: dict | FIBaseExtraInfo, info: ValidationInfo) -> FIBaseExtraInfo:
        if info.context:
            # multiplier = info.context.get('multiplier', 1)
            # value = value * multiplier
            pprint(info.context)
            cls_extra_info = info.context.get("cls_extra_info", FIBaseExtraInfo)
            print()
            print(cls_extra_info)
            pprint(value)
            print(type(value))
            loc_value = cls_extra_info(**value)
            return loc_value
        elif isinstance(value, dict):
            print("option B")
            loc_value = FIBaseExtraInfo(**value)
            return loc_value
        else:
            print("option C")
            return value