"""
Tests the basic type checking in serialize/deserialize
"""

import os
import sys
from pathlib import Path

import pytest

import pandas as pd
from pandas._testing import assert_frame_equal
import numpy as np


TESTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(TESTPATH, ".."))

import df_file_interchange as fi
from df_file_interchange.fi_generic import InvalidValueForFieldError
from df_file_interchange.fi_generic import _serialize_element, _deserialize_element


def test_serialize_uni8():
    candidate_elements = [-1, 256]

    for el in candidate_elements:
        with pytest.raises(InvalidValueForFieldError):
            dummy_result = _serialize_element(el, b_chk_correctness=True)


