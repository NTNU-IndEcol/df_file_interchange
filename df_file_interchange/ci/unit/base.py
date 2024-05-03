"""
Base unit definitions
"""

from typing import Literal

from pydantic import (
    BaseModel,
    computed_field,
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
    """

    # The unit description , e.g. "kg", "USD", "hamsters"
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
    """A generic unit, not sure using this now but will keep as placeholder for now..."""

    # Override so we can specify arbitrary strings
    unit_desc: Literal[None] | str = None
