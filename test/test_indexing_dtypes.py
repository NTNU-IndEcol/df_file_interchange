"""
Tests indexing and dtypes for df_file_interchange
"""

import os
import sys
from pathlib import Path

import pytest

import pandas as pd
import numpy as np

import df_file_interchange as fi

TESTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(TESTPATH, ".."))



@pytest.fixture()
def std_indices():
    return fi.fi_generic._generate_example_indices()


def test_save_load_indices(std_indices):
    dfs = fi.fi_generic._generate_dfs_from_indices(std_indices)
    pass
