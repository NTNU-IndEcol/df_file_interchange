"""
Tests indexing and dtypes for df_file_interchange
"""

from pathlib import Path
import pytest
import df_file_interchange as fi
from df_file_interchange.file.rw import chk_strict_frames_eq_ignore_nan


@pytest.fixture()
def std_indices():
    return fi.file.examples.generate_example_indices()


def test_save_load_indices(tmp_path: Path, std_indices):
    dfs = fi.file.examples.generate_dfs_from_indices(std_indices)

    for idx, df in dfs.items():
        print(f"Testing {idx}")

        # Generate and save CSV
        target_datafile_csv = tmp_path / f"test_df_{idx}__csv.csv"
        target_metafile_csv = tmp_path / f"test_df_{idx}__csv.yaml"
        metafile_csv = fi.write_df_to_file(
            df,
            target_datafile_csv,
            target_metafile_csv,
            fi.file.rw.FIFileFormatEnum.csv,
        )
        (df_reload_csv, metainfo_reload_csv) = fi.read_df(metafile_csv)

        # Generate and save parquet
        target_datafile_parquet = tmp_path / f"test_df_{idx}__parquet.parq"
        target_metafile_parquet = tmp_path / f"test_df_{idx}__parquet.yaml"
        metafile_parquet = fi.write_df_to_file(
            df,
            target_datafile_parquet,
            target_metafile_parquet,
            fi.file.rw.FIFileFormatEnum.parquet,
        )
        (df_reload_parquet, metainfo_reload_parquet) = fi.read_df(metafile_parquet)

        # Compare
        chk_strict_frames_eq_ignore_nan(
            df,
            df_reload_csv,
        )
        chk_strict_frames_eq_ignore_nan(
            df,
            df_reload_parquet,
        )

        # check_index_type=True,
        # check_column_type=True,
        # check_exact=True,
        # check_categorical=True,

        print(f"Done with {idx}")
