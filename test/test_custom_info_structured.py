"""
Tests structured custom metainfo handling/storage
"""

import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import pytest

TESTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(TESTPATH, ".."))

import df_file_interchange as fi
from df_file_interchange.file.rw import chk_strict_frames_eq_ignore_nan
from df_file_interchange.ci.examples import generate_example_with_metainfo_1
from df_file_interchange.ci.structured import generate_default_context


def test_save_load_examples(tmp_path: Path):

    # Get example dataframes
    (df1, custom_info1) = generate_example_with_metainfo_1()

    # Generate and save CSV
    target_datafile1_csv = tmp_path / f"test_df_example_1__csv.csv"
    target_metafile1_csv = tmp_path / f"test_df_example_1__csv.yaml"
    metafile1_csv = fi.write_df_to_file(
        df1,
        target_datafile1_csv,
        target_metafile1_csv,
        fi.FIFileFormatEnum.csv,
        custom_info=custom_info1,
    )
    (df1_reload_csv, metainfo1_reload_csv) = fi.read_df(
        metafile1_csv, context_metainfo=generate_default_context()
    )

    # Generate and save parquet
    target_datafile1_parquet = tmp_path / f"test_df_example_1__parquet.parq"
    target_metafile1_parquet = tmp_path / f"test_df_example_1__parquet.yaml"
    metafile1_parquet = fi.write_df_to_file(
        df1,
        target_datafile1_parquet,
        target_metafile1_parquet,
        fi.FIFileFormatEnum.parquet,
        custom_info=custom_info1,
    )
    (df1_reload_parquet, metainfo1_reload_parquet) = fi.read_df(
        metafile1_parquet, context_metainfo=generate_default_context()
    )

    # Compare dataframes
    chk_strict_frames_eq_ignore_nan(
        df1,
        df1_reload_csv,
    )
    chk_strict_frames_eq_ignore_nan(
        df1,
        df1_reload_parquet,
    )

    # Check custom info matches
    assert custom_info1 == metainfo1_reload_csv.custom_info
    assert custom_info1 == metainfo1_reload_parquet.custom_info
