"""
Structured custom info

Includes

* Column unit+descriptions
* Some basic optional info

"""

import traceback

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


# from ..file.rw import _init_context_var, init_context, FIBaseCustomInfo
from ..file.rw import FIBaseCustomInfo

from . import unit
from .unit.base import FIBaseUnit, FIGenericUnit
from .unit.currency import FICurrencyUnit


class FIBaseExtraInfo(BaseModel):

    # model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
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


class FIStdExtraInfo(FIBaseExtraInfo):

    author: str | None = None


class FIStructuredCustomInfo(FIBaseCustomInfo):
    """Structured custom info

    This includes extra data (applies to whole of table), and columnwise units.

    The extra info and column units can, however, be different classes, i.e. the
    user can inherit from `FIBaseExtraInfo` or `FIBaseUnit` to add more fields
    to extra info or define new units, respectively. This makes matters a little
    tricky when instantiating from metadata because we have to choose the
    correct class to create; in the normal run of things, this is not too bad
    but we allow userdefined classes presented at runtime. So this needs a bit
    of logic to get Pydantic to play ball. In particular, the user needs to
    supply a context with available classes to `model_validate()`.
    """

    # Extra meta info that applies to the whole table
    # Urgh: https://github.com/pydantic/pydantic/issues/7093
    extra_info: SerializeAsAny[FIBaseExtraInfo]

    # Columnwise unit info
    col_units: dict[Any, SerializeAsAny[FIBaseUnit]] = {}

    # def __init__(self, /, **data: Any) -> None:
    #     # Get parent to do its bit.
    #     super().__init__(**data)
    #     self.__pydantic_validator__.validate_python(
    #         data,
    #         self_instance=self,
    #         context=_init_context_var.get(),
    #     )


    @field_validator("extra_info", mode="before")
    @classmethod
    def validator_extra_info(
        cls, value: dict | FIBaseExtraInfo, info: ValidationInfo
    ) -> FIBaseExtraInfo:

        # Shortcut exit, if we've been passed something with extra_info already
        # instantiated. We only deal with dicts here.
        if not isinstance(value, dict):
            return value

        # If we don't have context, just use the base class or return as-is
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
        clss_extra_info = info.context.get(
            "clss_extra_info", {"FIBaseExtraInfo": FIBaseExtraInfo}
        )
        assert isinstance(clss_extra_info, dict)

        # Now process
        value_classname = value.get("classname", None)
        if value_classname and value_classname in clss_extra_info.keys():
            # return clss_extra_info[value_classname](**value)

            # Now instantiate the model
            extra_info_class = clss_extra_info[value_classname]
            assert isinstance(extra_info_class, FIBaseExtraInfo)
            return extra_info_class.model_validate(value, context=info.context)

        # Meh. Just use the base class, apparently we don't have a class
        # specified in the context for this.
        logger.warning(f"Missing context for extra_info deserialize. value={value}")
        return FIBaseExtraInfo.model_validate(value, context=info.context)

    @field_validator("col_units", mode="before")
    @classmethod
    def validator_col_units(
        cls, value: dict | FIBaseExtraInfo, info: ValidationInfo
    ) -> dict:

        # If this happens, we really need to fail.
        if not isinstance(value, dict):
            error_msg = f"col_units should always be a dictionary. Got type={type(value)}, value={value}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # If we don't have context, just return and let validation (likely)
        # fail.
        if not info.context:
            return value

        # Check our context is a dictionary
        assert isinstance(info.context, dict)

        # Get the available classes for units (this should also be a
        # dictionary)
        clss_col_units = info.context.get("clss_col_units", {"FIBaseUnit": FIBaseUnit})
        assert isinstance(clss_col_units, dict)

        # Now process each element in value, in turn
        loc_value = {}
        for idx in value:
            value_classname = value[idx].get("classname", None)
            if value_classname and value_classname in clss_col_units.keys():
                # loc_value[idx] = clss_col_units[value_classname](**(value[idx]))

                # Now instantiate the model and add to our local dictionary
                units_class = clss_col_units[value_classname]
                assert isinstance(units_class, FIBaseUnit)
                loc_value[idx] = units_class.model_validate(value[idx], context=info.context)
            else:
                warning_msg = f"Missing context for col_unit deserialize. idx={idx}, value[idx]={value[idx]}"
                logger.warning(warning_msg)

        return loc_value
