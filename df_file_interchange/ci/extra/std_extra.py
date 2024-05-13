"""Standard extra info"""

from loguru import logger  # noqa: F401

from datetime import date, datetime
from .base import FIBaseExtraInfo


class FIStdExtraInfo(FIBaseExtraInfo):
    """Standard extra info for a dataframe

    Attributes
    ----------
    author : str | None
        Default None.
    source : str | None
        Where it came from. Default None.
    """

    author: str | None = None
    source: str | None = None
    description: str | None = None
    processed_date: date | datetime | None = None
    processed_by: str | None = None
