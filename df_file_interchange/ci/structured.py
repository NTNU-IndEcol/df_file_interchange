"""
Structured custom info

Includes

* Column unit+descriptions
* Some basic optional info

"""

from typing import Any
from loguru import logger

# from contextlib import contextmanager
# from contextvars import ContextVar
from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    model_validator,
    field_validator,
    ValidationInfo,
    SerializeAsAny,
)


from .base import FIBaseCustomInfo
from .unit.base import FIBaseUnit
from .unit.currency import FICurrencyUnit
from .unit.population import FIPopulationUnit


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
    source: str | None = None


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

    @field_validator("extra_info", mode="before")
    @classmethod
    def validator_extra_info(
        cls, value: dict | FIBaseExtraInfo, info: ValidationInfo
    ) -> FIBaseExtraInfo:
        # Shortcut exit, if we've been passed something with extra_info already
        # instantiated. We only deal with dicts here.
        if not isinstance(value, dict):
            return value

        # Default don't use context
        clss_extra_info = None

        # Check if we've been supplied a context
        if info.context and isinstance(info.context, dict):
            # Get the available classes for extra_info (this should also be a
            # dictionary)
            clss_extra_info = info.context.get(
                "clss_extra_info", {"FIBaseExtraInfo": FIBaseExtraInfo}
            )
            assert isinstance(clss_extra_info, dict)

        # Now process
        value_classname = value.get("classname", None)
        if (
            value_classname
            and clss_extra_info is not None
            and value_classname in clss_extra_info.keys()
        ):
            # Now instantiate the model
            extra_info_class = clss_extra_info[value_classname]
        elif value_classname in globals().keys() and issubclass(
            globals()[value_classname], FIBaseExtraInfo
        ):
            extra_info_class = globals()[value_classname]
        else:
            error_msg = f"Neither context for supplied classname nor is it a subclass of FIBaseExtraInfo. classname={value_classname}"
            logger.error(error_msg)
            raise Exception(error_msg)

        assert issubclass(extra_info_class, FIBaseExtraInfo)
        return extra_info_class.model_validate(value, context=info.context)

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

        # By default we don't use a context
        clss_col_units = None

        # Check for a context
        if info.context and isinstance(info.context, dict):
            # Get the available classes for units (this should also be a
            # dictionary)
            clss_col_units = info.context.get(
                "clss_col_units", {"FIBaseUnit": FIBaseUnit}
            )
            assert isinstance(clss_col_units, dict)

        # Now process each element in value, in turn
        loc_value = {}
        for idx in value:
            # Skip if already instantiated
            if not isinstance(value[idx], dict):
                loc_value[idx] = value[idx]
                continue

            value_classname = value[idx].get("classname", None)
            if (
                value_classname
                and clss_col_units is not None
                and value_classname in clss_col_units.keys()
            ):
                # Now instantiate the model and add to our local dictionary
                units_class = clss_col_units[value_classname]
            elif value_classname in globals().keys() and issubclass(
                globals()[value_classname], FIBaseUnit
            ):
                units_class = globals()[value_classname]
            else:
                error_msg = f"Neither context supplied nor is subclass of FIBaseUnit. classname={value_classname}"
                logger.error(error_msg)
                raise Exception(error_msg)

            assert issubclass(units_class, FIBaseUnit)
            loc_value[idx] = units_class.model_validate(
                value[idx], context=info.context
            )

        return loc_value


def generate_default_context():
    """Generates a default context that is an 'all' for anything that is included by default

    Returns all available default units and FIStdExtraInfo.

    If you extend with your own units, you'll have to add manually.

    Returns
    -------
    dict
        The context.
    """

    context = {
        "clss_custom_info": {
            "FIStructuredCustomInfo": FIStructuredCustomInfo,
        },
        "clss_extra_info": {
            "FIStdExtraInfo": FIStdExtraInfo,
        },
        "clss_col_units": {
            "FICurrencyUnit": FICurrencyUnit,
            "FIPopulationUnit": FIPopulationUnit,
        },
    }

    return context
