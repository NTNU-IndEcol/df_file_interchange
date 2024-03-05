"""
Tests indexing and dtypes for df_file_interchange
"""

import os
import sys
from pathlib import Path

import pytest

import pandas as pd
import numpy as np


TESTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(TESTPATH, ".."))

import df_file_interchange as fi


@pytest.fixture()
def std_indices():
    return fi.fi_generic._generate_example_indices()


def test_save_load_indices(tmp_path: Path, std_indices):
    dfs = fi.fi_generic._generate_dfs_from_indices(std_indices)
    
    for idx, df in dfs.items():
        print(f"Testing {idx}")

        # Generate and save CSV
        target_datafile_csv = tmp_path / f"test_df_{idx}__csv.csv"
        target_metafile_csv = tmp_path / f"test_df_{idx}__csv.yaml"
        metafile_csv = fi.write_df_to_fi_generic(df, target_datafile_csv, target_metafile_csv, fi.FIFileFormatEnum.csv)
        (df_reload_csv, metainfo_reload_csv) = fi.read_fi_to_df_generic(metafile_csv)

        # Generate and save parquer
        target_datafile_parquet = tmp_path / f"test_df_{idx}__parquet.parq"
        target_metafile_parquet = tmp_path / f"test_df_{idx}__parquet.yaml"
        metafile_parquet = fi.write_df_to_fi_generic(df, target_datafile_parquet, target_metafile_parquet, fi.FIFileFormatEnum.parquet)
        (df_reload_parquet, metainfo_reload_parquet) = fi.read_fi_to_df_generic(metafile_parquet)

        # Compare
        pd._testing.assert_frame_equal(df, df_reload_csv)
        pd._testing.assert_frame_equal(df, df_reload_parquet)


