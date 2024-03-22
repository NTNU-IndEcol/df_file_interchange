"""
EXIOBASE/df_file_interchange
============================

Import a la ```import df_file_interchange as fi```.

"""

# We disable logging as default because we're only ever used as a library, see
# https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
from loguru import logger

logger.disable("df_file_interchange")

from .version import __version__

from .fi_generic import (
    FIFileFormatEnum,
    FIIndexType,
    FIEncodingCSV,
    FIEncodingParquet,
    FIEncoding,
    FIIndex,
    FIRangeIndex,
    FICategoricalIndex,
    FIMultiIndex,
    FIIntervalIndex,
    FIDatetimeIndex,
    FITimedeltaIndex,
    FIPeriodIndex,
    FIMetainfo,
    write_df_to_file,
    read_fi_to_df_generic,
)
