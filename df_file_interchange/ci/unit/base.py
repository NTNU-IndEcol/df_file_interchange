"""
Base unit definitions
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


class FIBaseUnit(BaseModel):
    """Base class for units

    You have to derive from this to define your own unit.

    Attributes
    ----------
    unit_desc : str literal
        For example, if the unit is a currency then this would be a literal of
        strs of the possible currencies. See `FICurrencyUnit` for this example.
    unit_multiplier : float
        Default 1.0. Used when, for example, we need to say "millions of (a
        unit)".
    unit_type : str (computed field)
        Override this in descendent classes to return what this is, e.g.
        "currency".
    """

    unit_desc: None = None

    # Sometimes we have quantities in "millions of $", for example
    unit_multiplier: int | float = 1.0

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


class FIGenericUnit(FIBaseUnit):

    # Override so we can specify arbitrary strings
    unit_desc: Literal[None] | str = None
