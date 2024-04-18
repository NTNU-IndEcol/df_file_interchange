"""
Column population units.

"""

from typing import Any, Literal, TypeAlias, Union
from loguru import logger

from pydantic import (
    BaseModel,
    computed_field,
)

from .base import FIBaseUnit


class FIPopulationUnit(FIBaseUnit):

    # The various currencies we can use.
    unit_desc: Literal[
        "people",
        "adults",
        "children",
        "pensioners",
        "women",
        "men",
    ]

    # This is probably what will be changed rather than unit_desc
    unit_multiplier: float = 1.0