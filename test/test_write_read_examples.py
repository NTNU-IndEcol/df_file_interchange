"""
Tests indexing and dtypes for df_file_interchange
"""

from pathlib import Path

import pytest  # noqa: F401
import pandas as pd
import numpy as np

import df_file_interchange as fi
from df_file_interchange.file.rw import chk_strict_frames_eq_ignore_nan


def test_write_read_function_args(tmp_path: Path):
    # Create a simple dataframe
    df = pd.DataFrame(np.random.randn(3, 4), columns=["a", "b", "c", "d"])

    # Save as CSV using `fi.write_df_to_csv()`
    target_datafile_csv_1 = tmp_path / "test_df_fn_args_1.csv"
    metafile_csv_1 = fi.write_df_to_csv(df, target_datafile_csv_1)
    (df_csv_1_reload, metainfo_csv_1_reload) = fi.read_df(metafile_csv_1)
    chk_strict_frames_eq_ignore_nan(
        df,
        df_csv_1_reload,
    )

    # Save as Parquet using `fi.write_df_to_parquet()`
    target_datafile_parq_1 = tmp_path / "test_df_fn_args_1.parq"
    metafile_parq_1 = fi.write_df_to_parquet(df, target_datafile_parq_1)
    (df_parq_1_reload, metainfo_parq_1_reload) = fi.read_df(metafile_parq_1)
    chk_strict_frames_eq_ignore_nan(
        df,
        df_parq_1_reload,
    )

    # Save to CSV using `fi.write_df_to_file()`, determine `file_format` automatically
    target_datafile_csv_2 = tmp_path / "test_df_fn_args_2.csv"
    target_metafile_csv_2 = tmp_path / "test_df_fn_args_2.yaml"
    metafile_csv_2 = fi.write_df_to_file(
        df, target_datafile_csv_2, target_metafile_csv_2
    )
    (df_csv_2_reload, metainfo_csv_2_reload) = fi.read_df(metafile_csv_2)
    chk_strict_frames_eq_ignore_nan(
        df,
        df_csv_2_reload,
    )

    # Save to Parquet using `fi.write_df_to_file()`, determine metafile and `file_format` automatically
    target_datafile_parq_2 = tmp_path / "test_df_fn_args_2.parq"
    metafile_parq_2 = fi.write_df_to_file(df, target_datafile_parq_2)
    (df_parq_2_reload, metainfo_parq_2_reload) = fi.read_df(metafile_parq_2)
    chk_strict_frames_eq_ignore_nan(
        df,
        df_parq_2_reload,
    )


def test_write_read_example_1(tmp_path: Path):
    # Get example dataframes
    df1 = fi.file.examples.generate_example_1()

    # Generate and save CSV, then compare
    target_datafile1_csv = tmp_path / "test_df_example_1__csv.csv"
    target_metafile1_csv = tmp_path / "test_df_example_1__csv.yaml"
    metafile1_csv = fi.write_df_to_file(
        df1, target_datafile1_csv, target_metafile1_csv, fi.file.rw.FIFileFormatEnum.csv
    )
    (df1_reload_csv, metainfo1_reload_csv) = fi.read_df(metafile1_csv)
    chk_strict_frames_eq_ignore_nan(
        df1,
        df1_reload_csv,
    )

    # Generate and save parquet, then compare
    target_datafile1_parquet = tmp_path / "test_df_example_1__parquet.parq"
    target_metafile1_parquet = tmp_path / "test_df_example_1__parquet.yaml"
    metafile1_parquet = fi.write_df_to_file(
        df1,
        target_datafile1_parquet,
        target_metafile1_parquet,
        fi.file.rw.FIFileFormatEnum.parquet,
    )
    (df1_reload_parquet, metainfo1_reload_parquet) = fi.read_df(metafile1_parquet)
    chk_strict_frames_eq_ignore_nan(
        df1,
        df1_reload_parquet,
    )
