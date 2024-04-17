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
    SerializeAsAny,
)


from ..file.rw import _init_context_var, init_context, FIBaseCustomInfo

from . import unit
from .unit.base import FIBaseUnit, FIGenericUnit
from .unit.currency import FICurrencyUnit


class FIBaseExtraInfo(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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

    @model_validator(mode='before')
    @classmethod
    def model_validator_extra_info(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "classname" in data.keys():
                del data["classname"]
        return data


class FIStdExtraInfo(FIBaseExtraInfo):

    author: str | None = None


class FIStructuredCustomInfo(FIBaseCustomInfo):

    # Extra meta info that applies to the whole table
    # Urgh: https://github.com/pydantic/pydantic/issues/7093
    extra_info: SerializeAsAny[FIBaseExtraInfo]

    # Columnwise unit info
    col_units: dict[Any, FIBaseUnit] = {}

    def __init__(self, /, **data: Any) -> None:

        pprint(data)

        # Get parent to do its bit.
        super().__init__(**data)

        self.__pydantic_validator__.validate_python(
            data,
            self_instance=self,
            context=_init_context_var.get(),
        )


    # This problably unnecessary now...
    # @model_serializer()
    # def serialize_model(self):

    #     # Need to get parent's serialization first
    #     loc_serialized = super().serialize_model()

    #     # Include extra_info and specify the class
    #     loc_serialized["extra_info"] = self.extra_info.model_dump()
    #     loc_serialized["_cls_extra_info"] = self.extra_info.__class__.__name__

    #     # Include col_units (we need to iterate through the dictionary)
    #     loc_serialized["col_units"] = {}
    #     loc_serialized["_cls_col_units"] = {}
    #     for idx in self.col_units:
    #         loc_serialized["col_units"][idx] = self.col_units[idx].model_dump()
    #         loc_serialized["_cls_col_units"][idx] = self.col_units[idx].__class__.__name__
        
    #     return loc_serialized
    

    @field_validator("extra_info", mode="before")
    @classmethod
    def validator_extra_info(
        cls, value: dict | FIBaseExtraInfo, info: ValidationInfo
    ) -> FIBaseExtraInfo:
        
        # Shortcut exit, if we've been passed something with extra_info already
        # instantiated. We only deal with dicts here.
        print("A")
        if not isinstance(value, dict):
            return value

        # If we don't have context, just use the base class or return as-is
        print("B")
        if not info.context:
            if isinstance(value, dict):
                loc_value = FIBaseExtraInfo(**value)
                return loc_value
            else:
                return value

        # Check our context is a dictionary
        assert isinstance(info.context, dict)                        

        # Get the available classes for extra_info (this should also be a
        # dictionary)
        print("C")
        clss_extra_info = info.context.get("clss_extra_info", {"FIBaseExtraInfo": FIBaseExtraInfo})
        assert isinstance(clss_extra_info, dict)
        
        # Now process
        print("D")
        value_classname = value.get("classname", None)
        if value_classname and value_classname in clss_extra_info.keys():
            # # Remove the classname field from the dictionary
            # del value["classname"]

            # Now instantiate the model
            return clss_extra_info[value_classname](**value)

        # Meh. Just use the base class, apparently we don't have a context for
        # this
        logger.warning(f"Missing context for extra_info deserialize. value={value}")
        return FIBaseExtraInfo(**value)





    # @field_validator("col_units", mode="before")
    # @classmethod
    # def validator_col_units(
    #     cls, value: dict, info: ValidationInfo
    # ) -> dict[Any, FIBaseUnit]:
        
    #     # N.B. The context only contains AVAILABLE units. We actually store the unit in the metainfo.
    #     # TODO document this

    #     # Shortcut return if user isn't specifying any units
    #     if not isinstance(value, dict) or len(value) == 0:
    #         return value

    #     loc_value = {}

    #     # Check if we have any context at all
    #     if info.context:
    #         ctx_units = info.context.get("_cls_col_units", {})
    #         if isinstance(ctx_units, dict):
    #             for idx in value:
    #                 # If not a dictionary at this entry, then we copy and skip
    #                 # processing... hope for the best.
    #                 if not isinstance(value[idx], dict):
    #                     loc_value[idx] = value[idx]
    #                     continue

    #                 # Retreive class name from the metainfo (if this doesn't
    #                 # exist, it's a failure)

    #                 # Chec, if not exists use generic
    #                 cls_unit = ctx_units.get(idx, FIGenericUnit)

    #                 # Convert this entry
    #                 loc_value[idx] = cls_unit(**value[idx])                    

    #             return loc_value

    #     # We don't have any context info
    #     for idx in value:
    #         # If not a dictionary at this entry, then we copy and skip
    #         # processing... hope for the best.
    #         if not isinstance(value[idx], dict):
    #             loc_value[idx] = value[idx]
    #             continue

    #         # Convert this key's dictionary to a generic unit type (because
    #         # we don't have context)
    #         loc_value[idx] = FIGenericUnit(**value[idx])

    #     return loc_value


