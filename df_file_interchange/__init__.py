"""
EXIOBASE/df_file_interchange
============================

Wrapper for Pandas to round-trip write+read to CSV/Parquet in a manner that
preserves indexes and dtypes, and allows storage of custom structured
meta-information.

Import in the style of ```import df_file_interchange as fi```.

"""

from loguru import logger
from . import util  # noqa: F401
from . import file  # noqa: F401

from .file.rw import (
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
