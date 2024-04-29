"""
EXIOBASE/df_file_interchange
============================

Import a la ```import df_file_interchange as fi```.

"""

from loguru import logger
from . import file  # noqa: F401

from .file.rw import (
    # FICategoricalIndex,
    # FIDatetimeIndex,
    # FIEncoding,
    # FIEncodingCSV,
    # FIEncodingParquet,
    # FIFileFormatEnum,
    # FIIndex,
    # FIIndexType,
    # FIIntervalIndex,
    # FIMetainfo,
    # FIMultiIndex,
    # FIPeriodIndex,
    # FIRangeIndex,
    # FITimedeltaIndex,
    chk_strict_frames_eq_ignore_nan,  # noqa: F401
    read_df,  # noqa: F401
    write_df_to_csv,  # noqa: F401
    write_df_to_file,  # noqa: F401
    write_df_to_parquet,  # noqa: F401
)

from . import ci  # noqa: F401
from .version import __version__  # noqa: F401


# We disable logging as default because we're only ever used as a library, see
# https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("df_file_interchange")
