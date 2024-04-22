"""
EXIOBASE/df_file_interchange
============================

Import a la ```import df_file_interchange as fi```.

"""

from loguru import logger

# We disable logging as default because we're only ever used as a library, see
# https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("df_file_interchange")

from . import file

from .file.rw import (
    FICategoricalIndex,
    FIDatetimeIndex,
    FIEncoding,
    FIEncodingCSV,
    FIEncodingParquet,
    FIFileFormatEnum,
    FIIndex,
    FIIndexType,
    FIIntervalIndex,
    FIMetainfo,
    FIMultiIndex,
    FIPeriodIndex,
    FIRangeIndex,
    FITimedeltaIndex,
    chk_strict_frames_eq_ignore_nan,
    read_df,
    write_df_to_csv,
    write_df_to_file,
    write_df_to_parquet,
)

# from . import deprecated_ci
from . import ci
from .ci.structured import generate_default_context
from .version import __version__
