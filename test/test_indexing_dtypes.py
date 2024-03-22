"""
Tests indexing and dtypes for df_file_interchange
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


@pytest.fixture()
def std_indices():
    return fi.fi_examples._generate_example_indices()


def test_save_load_indices(tmp_path: Path, std_indices):
    dfs = fi.fi_examples._generate_dfs_from_indices(std_indices)

    for idx, df in dfs.items():
        print(f"Testing {idx}")

        # Generate and save CSV
        target_datafile_csv = tmp_path / f"test_df_{idx}__csv.csv"
        target_metafile_csv = tmp_path / f"test_df_{idx}__csv.yaml"
        metafile_csv = fi.write_df_to_file(
            df, target_datafile_csv, target_metafile_csv, fi.FIFileFormatEnum.csv
        )
        (df_reload_csv, metainfo_reload_csv) = fi.read_df(metafile_csv)

        # Generate and save parquet
        target_datafile_parquet = tmp_path / f"test_df_{idx}__parquet.parq"
        target_metafile_parquet = tmp_path / f"test_df_{idx}__parquet.yaml"
        metafile_parquet = fi.write_df_to_file(
            df,
            target_datafile_parquet,
            target_metafile_parquet,
            fi.FIFileFormatEnum.parquet,
        )
        (df_reload_parquet, metainfo_reload_parquet) = fi.read_df(
            metafile_parquet
        )

        # Compare
        assert_frame_equal(df, df_reload_csv)
        assert_frame_equal(df, df_reload_parquet)

        print(f"Done with {idx}")


def test_save_load_examples(tmp_path: Path):

    # Get example dataframes
    df1 = fi.fi_examples._generate_example_1()

    # Generate and save CSV
    target_datafile1_csv = tmp_path / f"test_df_example_1__csv.csv"
    target_metafile1_csv = tmp_path / f"test_df_example_1__csv.yaml"
    metafile1_csv = fi.write_df_to_file(
        df1, target_datafile1_csv, target_metafile1_csv, fi.FIFileFormatEnum.csv
    )
    (df1_reload_csv, metainfo1_reload_csv) = fi.read_df(metafile1_csv)

    # Generate and save parquet
    target_datafile1_parquet = tmp_path / f"test_df_example_1__parquet.parq"
    target_metafile1_parquet = tmp_path / f"test_df_example_1__parquet.yaml"
    metafile1_parquet = fi.write_df_to_file(
        df1,
        target_datafile1_parquet,
        target_metafile1_parquet,
        fi.FIFileFormatEnum.parquet,
    )
    (df1_reload_parquet, metainfo1_reload_parquet) = fi.read_df(
        metafile1_parquet
    )

    # Compare
    assert_frame_equal(df1, df1_reload_csv)
    assert_frame_equal(df1, df1_reload_parquet)
