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
    model_serializer,
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

    author: str | None = None


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


    @model_serializer()
    def serialize_model(self):
        # Need to get parent's serialization first
        loc_serialized = super().model_dump()

        # Include extra_info and specify the class
        loc_serialized["extra_info"] = self.extra_info.model_dump()
        loc_serialized["_cls_extra_info"] = self.extra_info.__class__.__name__

        # Include col_units (we need to iterate through the dictionary)
        loc_serialized["col_units"] = {}
        loc_serialized["_cls_col_units"] = {}
        for idx in self.col_units:
            loc_serialized["col_units"][idx] = self.col_units[idx].model_dump()
            loc_serialized["_cls_col_units"][idx] = self.col_units[idx].__class__.__name__
        
        return loc_serialized
    

    @field_validator("extra_info", mode="before")
    @classmethod
    def validator_extra_info(
        cls, value: dict | FIBaseExtraInfo, info: ValidationInfo
    ) -> FIBaseExtraInfo:
        if info.context:
            assert isinstance(info.context, dict)
            cls_extra_info = info.context.get("cls_extra_info", FIBaseExtraInfo)
            if isinstance(value, dict):
                loc_value = cls_extra_info(**value)
                return loc_value
            else:
                if not isinstance(value, cls_extra_info):
                    raise NotImplementedError(
                        "Haven't implemented code to handle this situation yet (being passed a FBaseExtraInfo class/descendent that is different from the context)"
                    )
                return value
        else:
            if isinstance(value, dict):
                loc_value = FIBaseExtraInfo(**value)
                return loc_value
            else:
                return value



    # @field_validator("extra_info", mode="before")
    # @classmethod
    # def validator_extra_info(
    #     cls, value: dict | FIBaseExtraInfo, info: ValidationInfo
    # ) -> FIBaseExtraInfo:
    #     if info.context:
    #         assert isinstance(info.context, dict)
    #         cls_extra_info = info.context.get("cls_extra_info", FIBaseExtraInfo)
    #         if isinstance(value, dict):
    #             loc_value = cls_extra_info(**value)
    #             return loc_value
    #         else:
    #             if not isinstance(value, cls_extra_info):
    #                 raise NotImplementedError(
    #                     "Haven't implemented code to handle this situation yet (being passed a FBaseExtraInfo class/descendent that is different from the context)"
    #                 )
    #             return value
    #     else:
    #         if isinstance(value, dict):
    #             loc_value = FIBaseExtraInfo(**value)
    #             return loc_value
    #         else:
    #             return value


