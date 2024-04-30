"""Examples with metainfo, mainly used for tests"""

import numpy as np
import pandas as pd


from .extra.std_extra import FIStdExtraInfo
from .structured import FIStructuredCustomInfo

from .unit.currency import FICurrencyUnit
from .unit.population import FIPopulationUnit


def generate_example_with_metainfo_1():
    """Generates a sample dataframe with custom info

    Returns
    -------
    tuple
        (df, custom_info)
    """

    # Create basic dataframe
    df = pd.DataFrame(np.random.randn(3, 4), columns=["a", "b", "c", "d"])
    df["pop"] = pd.array([1234, 5678, 91011])

    unit_cur_a = FICurrencyUnit(unit_desc="USD", unit_multiplier=1000)
    unit_cur_b = FICurrencyUnit(unit_desc="EUR", unit_multiplier=1000)
    unit_cur_c = FICurrencyUnit(unit_desc="JPY", unit_multiplier=1000000)
    unit_cur_d = FICurrencyUnit(unit_desc="USD", unit_multiplier=1000)
    unit_pop = FIPopulationUnit(unit_desc="people", unit_multiplier=1)

    extra_info = FIStdExtraInfo(author="Spud", source="Potato")

    custom_info = FIStructuredCustomInfo(
        extra_info=extra_info,
        col_units={
            "a": unit_cur_a,
            "b": unit_cur_b,
            "c": unit_cur_c,
            "d": unit_cur_d,
            "pop": unit_pop,
        },
    )

    return (df, custom_info)
