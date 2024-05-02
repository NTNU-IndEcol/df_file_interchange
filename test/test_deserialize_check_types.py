"""
Tests the basic type checking in deserialize
"""

import numpy as np
import pytest


from df_file_interchange.file.rw import (
    InvalidValueForFieldError,
    _deserialize_element,
)

badness_int_list = [
    (np.iinfo(np.uint8).min - 1, "np.uint8"),
    (np.iinfo(np.uint16).min - 1, "np.uint16"),
    (np.iinfo(np.uint32).min - 1, "np.uint32"),
    (np.iinfo(np.uint64).min - 1, "np.uint64"),
    (np.iinfo(np.ulonglong).min - 1, "np.ulonglong"),
    (np.iinfo(np.uint8).max + 1, "np.uint8"),
    (np.iinfo(np.uint16).max + 1, "np.uint16"),
    (np.iinfo(np.uint32).max + 1, "np.uint32"),
    (np.iinfo(np.uint64).max + 1, "np.uint64"),
    (np.iinfo(np.ulonglong).max + 1, "np.ulonglong"),
    (1.123, "np.uint8"),
    (1.123, "np.uint16"),
    (1.123, "np.uint32"),
    (1.123, "np.uint64"),
    (1.123, "np.ulonglong"),
    (np.nan, "np.uint8"),
    (np.nan, "np.uint16"),
    (np.nan, "np.uint32"),
    (np.nan, "np.uint64"),
    (np.nan, "np.ulonglong"),
    (np.inf, "np.uint8"),
    (np.inf, "np.uint16"),
    (np.inf, "np.uint32"),
    (np.inf, "np.uint64"),
    (np.inf, "np.ulonglong"),
    (np.iinfo(np.int8).min - 1, "np.int8"),
    (np.iinfo(np.int16).min - 1, "np.int16"),
    (np.iinfo(np.int32).min - 1, "np.int32"),
    (np.iinfo(np.int64).min - 1, "np.int64"),
    (np.iinfo(np.longlong).min - 1, "np.longlong"),
    (np.iinfo(np.int8).max + 1, "np.int8"),
    (np.iinfo(np.int16).max + 1, "np.int16"),
    (np.iinfo(np.int32).max + 1, "np.int32"),
    (np.iinfo(np.int64).max + 1, "np.int64"),
    (np.iinfo(np.longlong).max + 1, "np.longlong"),
    (1.123, "np.int8"),
    (1.123, "np.int16"),
    (1.123, "np.int32"),
    (1.123, "np.int64"),
    (1.123, "np.longlong"),
    (np.nan, "np.int8"),
    (np.nan, "np.int16"),
    (np.nan, "np.int32"),
    (np.nan, "np.int64"),
    (np.nan, "np.longlong"),
    (np.inf, "np.int8"),
    (np.inf, "np.int16"),
    (np.inf, "np.int32"),
    (np.inf, "np.int64"),
    (np.inf, "np.longlong"),
]


@pytest.mark.parametrize("el,dtype", badness_int_list)
def test_badness_deserialize_np_ints(el, dtype):
    with pytest.raises(InvalidValueForFieldError):
        dummy_result = _deserialize_element(  # noqa: F841
            {"el": el, "eltype": dtype}, b_chk_correctness=True
        )
