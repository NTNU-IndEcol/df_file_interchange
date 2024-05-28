"""
Column population units.

"""

from typing import Literal

from .base import FIBaseUnit


class FIPopulationUnit(FIBaseUnit):
    """(Pydantic) class that defines population as a unit

    Includes possibility to define not only as persons but as 'adults',
    'children', 'women', 'men', etc.

    Attributes
    ----------
    unit_desc : Literal
        Whether we're talking about 'people', 'adults', etc.

    unit_multiplier : int
        How many of `unit_desc` at one time, e.g. 1_000_000 persons. Default 1.
    """

    # The various currencies we can use.
    unit_desc: Literal[
        "people",
        "adults",
        "children",
        "pensioners",
        "women",
        "men",
    ] = "people"

    # This is probably what will be changed rather than unit_desc
    unit_multiplier: int = 1
