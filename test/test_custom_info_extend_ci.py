"""
Tests structured custom metainfo handling/storage
"""

from typing import Literal
from pathlib import Path

# TESTPATH = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(TESTPATH, ".."))

import pandas as pd
import numpy as np
import df_file_interchange as fi
from df_file_interchange.file.rw import chk_strict_frames_eq_ignore_nan
from df_file_interchange.ci.unit.base import FIBaseUnit
from df_file_interchange.ci.extra.std_extra import FIStdExtraInfo
from df_file_interchange.ci.structured import FIStructuredCustomInfo


class FITestUnit(FIBaseUnit):
    unit_desc: str = ""
    hamster_type: Literal["syrian", "space-hamster"] = "syrian"


class FITestExtraInfo(FIStdExtraInfo):
    hamster_collective: str | None = None


class FITestStructuredCustomInfo(FIStructuredCustomInfo):
    hamster_motto: str | None = None


def test_save_load_with_extend_extra_info(tmp_path: Path):
    # Create dataframe with extended custom info classes
    df = pd.DataFrame(
        np.random.randn(4, 3),
        columns=["pet hamsters", "wild hamsters", "space hamsters"],
    )

    unit_cur_pet = FITestUnit(unit_desc="hamster", hamster_type="syrian")
    unit_cur_wild = FITestUnit(unit_desc="hamster", hamster_type="syrian")
    unit_cur_space = FITestUnit(unit_desc="hamster", hamster_type="space-hamster")

    extra_info = FITestExtraInfo(
        author="Spud", source="Potato", hamster_collective="going nuts"
    )

    custom_info = FITestStructuredCustomInfo(
        extra_info=extra_info,
        col_units={
            "unit_cur_pet": unit_cur_pet,
            "unit_cur_wild": unit_cur_wild,
            "unit_cur_space": unit_cur_space,
        },
        hamster_motto="jump, baby",
    )

    # Save CSV
    target_datafile_csv = tmp_path / "test_df_extend_ci_1__csv.csv"
    metafile = fi.write_df_to_csv(
        df,
        target_datafile_csv,
        custom_info=custom_info,
    )

    # Define necessary context
    context = {
        "clss_custom_info": {
            "FITestStructuredCustomInfo": FITestStructuredCustomInfo,
        },
        "clss_extra_info": {
            "FITestExtraInfo": FITestExtraInfo,
        },
        "clss_col_units": {
            "FITestUnit": FITestUnit,
        },
    }

    # Read
    (df_reload, metainfo_reload) = fi.read_df(metafile, context_metainfo=context)

    # Compare dataframes
    chk_strict_frames_eq_ignore_nan(
        df,
        df_reload,
    )

    # Check classes are correct
    assert isinstance(metainfo_reload.custom_info, FITestStructuredCustomInfo)
    assert isinstance(metainfo_reload.custom_info.extra_info, FITestExtraInfo)
    assert isinstance(metainfo_reload.custom_info.col_units["unit_cur_pet"], FITestUnit)
    assert isinstance(
        metainfo_reload.custom_info.col_units["unit_cur_wild"], FITestUnit
    )
    assert isinstance(
        metainfo_reload.custom_info.col_units["unit_cur_space"], FITestUnit
    )
