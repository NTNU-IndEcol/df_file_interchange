"""
Column population units.

"""

from typing import Literal


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
    unit_multiplier: int = 1
