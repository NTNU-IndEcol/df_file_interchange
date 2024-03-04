"""
Generic version of the File Interchange code.

KST requested that this code should be generic. So it's agnostic in the sense
that it saves/loads according to predefined Pydantic Models or dictionaries that
are supplied. There are no imports used from the rest of datamanager.

The datamanager layer on top of this is in fi_dm.py.

Note to self: make sure that functionality is split into methods so that can
override in descendent classes, e.g. processing paths from the metafile pointing
to the data file should be overridable according to datamanager semantics at a
later point.

Warning: We set CoW semantics (will be in Pandas 3.0 anyway)!

Note to self:

* pandas.Float64Dtype converts np.NaN and np.Inf to <NA>.

Pandas issues:

* Generally unable to use to_json() with the dtype: it's broken in several places.
* Weird bug when creating dataframe with single categorical column.
* There's issues with ordering when using CategoricalIndex in a Multiindex. See
  https://stackoverflow.com/questions/71837659/trying-to-sort-multiindex-index-using-categorical-index
  and https://github.com/pandas-dev/pandas/issues/47607

Pandas sorted issues:

* assert_frame_equal() fails when it shouldn't. I'm really tired of this
  nonsense now. See ttps://github.com/pandas-dev/pandas/issues/57644


Discussion:

Code must be able to write out and read in exactly the same dataframe. This
turns out to be a fair but more tricky with Pandas than it should be: they've
been too smart for their own good, and there's a few bugs in annoying places. So
instead of an elegant solution, have had to manually write code to properly
serialise column dtype information.

"""

import csv
import numpy as np
import pandas as pd
import copy
import yaml
import json
import hashlib, hmac

from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import (
    Any,
    Literal,
    Union,
    TypeAlias,
)

from datetime import (
    tzinfo,
)


from pandas._libs.tslibs import BaseOffset

# Pylance complains about Frequency, presumably because it uses a ForwardRef.
# It's not striclty wrong but will redefine locally for now: see _Frequency
# later on.
# from pandas._typing import ArrayLike, AnyArrayLike, Frequency
from pandas._typing import Dtype, ArrayLike, AnyArrayLike, IntervalClosedType, DtypeObj

# DO NOT try to remove these. It's required by Pydantic to resolve forward
# references, I think. Anyway, it complains without this.
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas import Index, Series

from loguru import logger


from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
    computed_field,
    field_serializer,
)
from typing_extensions import Annotated
from zoneinfo import ZoneInfo

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# Set CoW semantics
pd.set_option("mode.copy_on_write", True)
pd.options.mode.copy_on_write = True

# TODO this isn't nice. Pylance complains about
_Frequency = Union[str, BaseOffset]

# Our "anything like a list"
TArrayThing: TypeAlias = list | tuple | np.ndarray


# def save(df, author, url, source, year, country):
#     df.to_xxx
#     **kwars -> yaml
#     ** pandas_file_format -> yaml
# a, b, c, d
# a ... mean gdp
# b ... mean gdp per capita
# def load(path):
#     reads yaml
#     select from_xxx from yaml, with paramters from yaml - file_paramters
#     meta_dict = yaml (except file_paramters)
#     return (df, meta_dict)
# yaml metadata file : metadata.yaml
# file_paramters:  - AUTOMATIC
#     type: parquet, csv, etc.
#     index_col
#     headers
#     sep:
#     encoding:
# general:  - USER MANUAL ENTRY, partly optional
#     source
#     version
#     author
#     country
#     currency
# columns:  - USER MANUAL, fully optional?
#     for each column:
#     column_name: description


def str_n(in_str):
    """Does a simple str cast but if None converts to empty str

    Parameters
    ----------
    in_str : Any

    Returns
    -------
    str
        str(in_str) or "" if in_str == None
    """

    if in_str is None:
        return ""
    else:
        return str(in_str)


def _serialize_element(el) -> dict:
    """Serialize something

    Parameters
    ----------
    el : list, tuple, array, int, str, etc.

    Returns
    -------
    dict
        With elements `el` with the serialized content and `eltype` to describe
        how to deserialize.
    """

    # TODO recursion protection
    if isinstance(el, list):
        loc_el = _serialize_list_with_types(el)
        loc_type = "list"
    elif isinstance(el, tuple):
        loc_el = _serialize_list_with_types(list(el))
        loc_type = "tuple"
    elif isinstance(el, np.ndarray):
        assert el.ndim == 1
        loc_el = _serialize_list_with_types(list(el))
        loc_type = "np.ndarray"
    elif isinstance(el, pd.Index):
        # This isn't idea since it's replicating functionality that's in FIIndex
        # but we're only using a subset of that here and it's only for use with
        # MultiIndex encoding.
        loc_el = {
            "dtype": str(el.dtype),
            "elements": _serialize_list_with_types(list(el)),
        }
        loc_type = "pd.Index"
    elif isinstance(el, pd.arrays.DatetimeArray):
        loc_el = {
            "dtype": str(el.dtype),
            "elements": _serialize_list_with_types(list(el)),
        }
        loc_type = "pd.arrays.DatetimeArray"
    elif isinstance(el, pd.arrays.PeriodArray):
        loc_el = {
            "dtype": str(el.dtype),
            "elements": _serialize_list_with_types(list(el)),
        }
        loc_type = "pd.arrays.PeriodArray"
    elif isinstance(el, pd.Timestamp):
        loc_el = {"isoformat": str(el.isoformat()), "tz": str_n(el.tz)}
        loc_type = "pd.Timestamp"
    elif isinstance(el, np.datetime64):
        loc_el = str(el)
        loc_type = "np.datetime64"
    elif isinstance(el, pd.Interval):
        serialized_left = _serialize_element(el.left)
        serialized_right = _serialize_element(el.right)
        loc_el = {
            "left": serialized_left,
            "right": serialized_right,
            "closed": el.closed,
        }
        loc_type = "pd.Interval"
    elif isinstance(el, pd.Period):
        loc_el = str(el)
        loc_type = "pd.Period"
    elif isinstance(el, str):
        loc_el = el
        loc_type = "str"
    elif isinstance(el, int):
        loc_el = el
        loc_type = "int"
    elif isinstance(el, float):
        loc_el = el
        loc_type = "float"
    elif type(el) == np.int8:
        loc_el = int(el)
        loc_type = "np.int8"
    elif type(el) == np.int16:
        loc_el = int(el)
        loc_type = "np.int16"
    elif type(el) == np.int32:
        loc_el = int(el)
        loc_type = "np.int32"
    elif type(el) == np.int64:
        loc_el = int(el)
        loc_type = "np.int64"
    elif type(el) == np.longlong:
        loc_el = int(el)
        loc_type = "np.longlong"
    elif type(el) == np.uint8:
        loc_el = int(el)    # DO NOT use abs() here as we don't want to modify (error would be detected on deserialize)
        loc_type = "np.uint8"
    elif type(el) == np.uint16:
        loc_el = int(el)    # DO NOT use abs() here as we don't want to modify (error would be detected on deserialize)
        loc_type = "np.uint16"
    elif type(el) == np.uint32:
        loc_el = int(el)    # DO NOT use abs() here as we don't want to modify (error would be detected on deserialize)
        loc_type = "np.uint32"
    elif type(el) == np.uint64:
        loc_el = int(el)    # DO NOT use abs() here as we don't want to modify (error would be detected on deserialize)
        loc_type = "np.uint64"
    elif type(el) == np.ulonglong:
        loc_el = int(el)
        loc_type = "np.ulonglong"
    elif type(el) == np.float16:
        loc_el = float(el)
        loc_type = "np.float16"
    elif type(el) == np.float32:
        loc_el = float(el)
        loc_type = "np.float32"
    elif type(el) == np.float64:
        loc_el = float(el)
        loc_type = "np.float64"
    elif type(el) == np.complex64:
        loc_el = complex(el)
        loc_type = "np.complex64"
    elif type(el) == np.complex64:
        loc_el = complex(el)
        loc_type = "np.complex128"
    elif type(el) == np.clongdouble:
        loc_el = complex(el)
        loc_type = "np.clongdouble"
    else:
        warning_msg = (
            f"Serializing element got unexpected type: el={el}, type={type(el)}"
        )
        logger.warning(warning_msg)
        loc_el = el
        loc_type = ""

    return {"el": loc_el, "eltype": loc_type}


def _deserialize_element(serialized_element: dict):
    """Deserialize a dict created by `_serialize_element()`

    Parameters
    ----------
    serialized_element : dict

    """

    # TODO field checks and recursion protection

    el = serialized_element["el"]
    eltype = serialized_element["eltype"]

    if eltype == "list":
        return _deserialize_list_with_types(el)
    elif eltype == "tuple":
        return tuple(_deserialize_list_with_types(el))
    elif eltype == "np.ndarray":
        return np.array(_deserialize_list_with_types(el))
    elif eltype == "pd.Index":
        return pd.Index(
            _deserialize_list_with_types(el["elements"]), dtype=el["dtype"], copy=True
        )
    elif eltype == "pd.arrays.DatetimeArray":
        return pd.arrays.DatetimeArray._from_sequence(_deserialize_list_with_types(el["elements"]), dtype=el["dtype"], copy=True)  # type: ignore
    elif eltype == "pd.arrays.PeriodArray":
        return pd.arrays.PeriodArray._from_sequence(_deserialize_list_with_types(el["elements"]), dtype=el["dtype"], copy=True)  # type: ignore
    elif eltype == "pd.Timestamp":
        if el["tz"] == "":
            return pd.Timestamp(el["isoformat"])
        else:
            return pd.Timestamp(el["isoformat"], tz=el["tz"])
    elif eltype == "np.datetime64":
        return np.datetime64(el)
    elif eltype == "pd.Interval":
        deserialize_left = _deserialize_element(el["left"])
        deserialize_right = _deserialize_element(el["right"])
        return pd.Interval(
            left=deserialize_left, right=deserialize_right, closed=el["closed"]
        )
    elif eltype == "pd.Period":
        return pd.Period(el)
    elif eltype == "str":
        return str(el)
    elif eltype == "int":
        return int(el)
    elif eltype == "float":
        return float(el)
    elif eltype == "np.int8":
        return np.int8(el)
    elif eltype == "np.int16":
        return np.int16(el)
    elif eltype == "np.int32":
        return np.int32(el)
    elif eltype == "np.int64":
        return np.int64(el)
    elif eltype == "np.longong":
        return np.longlong(el)
    elif eltype == "np.uint8":
        return np.uint8(el)
    elif eltype == "np.uint16":
        return np.uint16(el)
    elif eltype == "np.uint32":
        return np.uint32(el)
    elif eltype == "np.uint64":
        return np.uint64(el)
    elif eltype == "np.ulongong":
        return np.ulonglong(el)
    elif eltype == "np.float16":
        return np.float16(el)
    elif eltype == "np.float32":
        return np.float32(el)
    elif eltype == "np.float64":
        return np.float64(el)
    elif eltype == "np.complex64":
        return np.complex64(el)
    elif eltype == "np.complex128":
        return np.complex128(el)
    elif eltype == "np.clongdouble":
        return np.clongdouble(el)
    
    else:
        return el


def _serialize_list_with_types(data: list) -> list:
    """Serialize a list (don't call this directly, use `_serialize_element()` instead)

    Serializes list elementwise.

    Parameters
    ----------
    data : list

    Returns
    -------
    list
    """

    loc_data = []
    for item in data:
        loc_data.append(_serialize_element(item))

    return loc_data


def _deserialize_list_with_types(serialized_data: list[dict]) -> list:

    loc_data = []
    for item in serialized_data:
        assert isinstance(item, dict)
        assert "el" in item.keys()
        assert "eltype" in item.keys()
        loc_data.append(_deserialize_element(item))

    return loc_data


class FIFileFormatEnum(str, Enum):
    """File formats used by file interchange"""

    csv = "csv"
    parquet = "parquet"


class FIIndexType(str, Enum):
    """The type of an index, e.g. RangeIndex, Categorical, MultiIndex"""

    base = "base"
    idx = "idx"  # Using literal "index" seems to cause a problem.
    range = "range"
    categorical = "categorical"
    multi = "multi"
    interval = "interval"
    datetime = "datetime"
    timedelta = "timedelta"
    period = "period"


class FIEncodingCSV(BaseModel):
    """The parameters we use for writing or reading CSV files.

    NOTE! You almost certainly do not have any reason to change these defaults.
    They were tested to ensure that the roundtrip write-read is exactly correct.

    Attributes
    ----------

    """

    # WE write all our files, so we can be more restrictive to reduce window for
    # ambiguity when reading a file. In particule, it's a bad idea to confuse
    # NaN with a null value with a missing value with an empty value -- these
    # are NOT the same, despite what "data science" conventions might suggest.
    # If you must be awkward, try
    #   ["-NaN", "-nan", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"]
    #   noting that "" is not in that list (that does cause problems).
    csv_allowed_na: list[str] = ["<NA>"]

    # # We write headers and indexes
    # header: bool = True
    # index: bool = True

    # Explictly define field separator
    sep: str = ","

    # This must be in the csv_allowed_na list
    na_rep: str = "<NA>"
    keep_default_na: bool = False

    # How we're escaping quotes in a str
    doublequote: bool = True

    # We only quote non-numeric values
    quoting: int = csv.QUOTE_NONNUMERIC

    # Weirdly, Pandas's other options, including the default, don't actually
    # return what was written with floats.
    float_precision: Literal["high", "legacy", "round_trip"] = "round_trip"

    # Do logic checks now that we've got the fields sorted
    @model_validator(mode="after")
    # def check_logic(self) -> "FIEncodingCSV":     # -- removed return type to fix warnings
    def check_logic(self):
        # Check if oid set then must be in correct format
        if self.na_rep != "":
            if not self.na_rep in self.csv_allowed_na:
                error_msg = (
                    f"na_rep must be in csv_allowed_na. na_rep={self.na_rep};"
                    f" csv_allowed_na={self.csv_allowed_na}"
                )
                logger.error(error_msg)
                raise Exception(error_msg)

        return self


class FIEncodingParquet(BaseModel):
    """The parameters we used for writing parquet files"""

    # Engine to use. Has to be consistent and was tested with pyarrow
    engine: str = "pyarrow"

    # Needs to be True. If None (default), Pandas will store a RangeIndex in the
    # Parquet meta, which makes it harder to import from other software. See
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html#pandas.DataFrame.to_parquet
    index: str | None = None


class FIEncoding(BaseModel):
    """General encoding options, includes CSV and Parquet encoding"""

    # Extra options that depend on format
    csv: FIEncodingCSV = FIEncodingCSV()
    parq: FIEncodingParquet = FIEncodingParquet()

    # Whether to automatically convert standard int dtypes to Pandas's
    # Int64Dtype (which can also encode NA values), if there are one or more NAs
    # or None(s) in the column
    auto_convert_int_to_intna: bool = True


class FIBaseIndex(BaseModel):
    """Base class for defining our custom classes to be able to
    serialize/deserialize/instantiate Pandas indexes"""

    # TODO factory code to instantiate itself? (if possible from Pydantic model)

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        return FIIndexType.base.name

    def get_fi_index_type(self) -> FIIndexType:
        return FIIndexType.base

    def get_as_index(self, **kwargs) -> pd.Index:
        return pd.Index()


class FIIndex(FIBaseIndex):
    """Corresonds to pd.Index"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: ArrayLike | AnyArrayLike | list | tuple
    name: str | None = None
    dtype: Dtype | DtypeObj | pd.api.extensions.ExtensionDtype | None

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        return FIIndexType.idx.name

    def get_fi_index_type(self) -> str:
        return FIIndexType.idx

    def get_as_index(self, **kwargs) -> pd.Index:
        return pd.Index(
            data=self.data,
            name=self.name,
            dtype=self.dtype,
            copy=True,
        )

    @field_serializer("data", when_used="always")
    def serialize_data(self, data: ArrayLike | AnyArrayLike | list | tuple):
        return _serialize_element(list(data))

    @field_serializer("dtype", when_used="always")
    def serialize_index_type(self, dtype: Dtype | None):
        return str(dtype)

    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check if data provided is a "true" data array or if it's serialized from before
            # if "data" in data.keys() and len(data["data"]) > 0 and isinstance(data["data"][0], dict) and "eltype" in data["data"][0].keys():
            if (
                "data" in data.keys()
                and isinstance(data["data"], dict)
                and "el" in data["data"].keys()
                and "eltype" in data["data"].keys()
            ):
                data["data"] = _deserialize_element(data["data"])

        return data


class FIRangeIndex(FIBaseIndex):
    """Corresonds to pd.RangeIndex"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: int
    stop: int
    step: int
    name: str | None = None
    dtype: DtypeObj | pd.api.extensions.ExtensionDtype | str | None

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        return FIIndexType.range.name

    def get_fi_index_type(self) -> str:
        return FIIndexType.range

    def get_as_index(self, **kwargs) -> pd.RangeIndex:
        return pd.RangeIndex(
            start=self.start,
            stop=self.stop,
            step=self.step,
            name=self.name,
            dtype=self.dtype,
        )

    @field_serializer("dtype", when_used="always")
    def serialize_dtype(self, dtype: Dtype | None):
        return str(dtype)


class FICategoricalIndex(FIBaseIndex):
    """Corresonds to pd.CategoricalIndex"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: ArrayLike | AnyArrayLike | list | tuple
    categories: ArrayLike | AnyArrayLike | list | tuple
    ordered: bool
    name: str | None = None
    dtype: (
        DtypeObj | pd.api.extensions.ExtensionDtype | pd.CategoricalDtype | str | None
    )

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        return FIIndexType.categorical.name

    def get_fi_index_type(self) -> str:
        return FIIndexType.categorical

    def get_as_index(self, **kwargs) -> pd.CategoricalIndex:
        return pd.CategoricalIndex(
            data=self.data,
            categories=self.categories,
            ordered=self.ordered,
            name=self.name,
            dtype=self.dtype,
            copy=True,
        )

    @field_serializer("data", when_used="always")
    def serialize_data(self, data: ArrayLike | AnyArrayLike | list | tuple):
        return _serialize_element(list(data))

    @field_serializer("categories", when_used="always")
    def serialize_categories(self, categories: ArrayLike | AnyArrayLike | list | tuple):
        return _serialize_element(list(categories))

    @field_serializer("dtype", when_used="always")
    def serialize_dtype(self, dtype: Dtype | None):
        return str(dtype)

    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # if "data" in data.keys() and len(data["data"]) > 0 and isinstance(data["data"][0], dict) and "eltype" in data["data"][0].keys():
            if (
                "data" in data.keys()
                and isinstance(data["data"], dict)
                and "el" in data["data"].keys()
                and "eltype" in data["data"].keys()
            ):
                data["data"] = _deserialize_element(data["data"])

            # if "categories" in data.keys() and len(data["categories"]) > 0 and isinstance(data["categories"][0], dict) and "eltype" in data["categories"][0].keys():
            if (
                "categories" in data.keys()
                and isinstance(data["categories"], dict)
                and "el" in data["categories"].keys()
                and "eltype" in data["categories"].keys()
            ):
                data["categories"] = _deserialize_element(data["categories"])

        return data


class FIMultiIndex(FIBaseIndex):
    """Corresponds to pd.MultiIndex"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    levels: list
    codes: list
    sortorder: int | None = None
    names: list
    dtypes: pd.Series | list  # Hmmm.

    # Need some extra validation logic to ensure FrozenList(s) contain what is expected

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        return FIIndexType.multi.name

    def get_fi_index_type(self) -> str:
        return FIIndexType.multi

    def get_as_index(self, **kwargs) -> pd.MultiIndex:
        return pd.MultiIndex(
            levels=self.levels,
            codes=self.codes,
            sortorder=self.sortorder,
            names=self.names,
            dtype=self.dtypes,  # Not used in Pandas source
            copy=True,
            verify_integrity=True,
        )

    @field_serializer("levels", when_used="always")
    def serialize_levels(self, levels: list):
        loc_levels = []
        for level in levels:
            loc_levels.append(_serialize_element(level))

        return loc_levels

    @field_serializer("codes", when_used="always")
    def serialize_codes(self, codes: list):
        loc_codes = []
        for code in codes:
            loc_codes.append(_serialize_element(code))

        return loc_codes

    @field_serializer("names", when_used="always")
    def serialize_names(self, names: list):
        if isinstance(names, np.ndarray):
            return names.tolist()
        else:
            return list(names)

    @field_serializer("dtypes", when_used="always")
    def serialize_dtypes(self, dtypes: pd.Series | list):
        # Ouch.
        return list(map(str, list(dtypes)))

    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data: Any) -> Any:
        if isinstance(data, dict):

            # Check if data provided is a "true" data array or if it's serialized from before
            if (
                "levels" in data.keys()
                and len(data["levels"]) > 0
                and isinstance(data["levels"], list)
            ):
                loc_levels = []
                for cur_level in data["levels"]:
                    # Need to test whether we're deserializing or de novo construction
                    if (
                        isinstance(cur_level, dict)
                        and "el" in cur_level.keys()
                        and "eltype" in cur_level.keys()
                    ):
                        loc_levels.append(_deserialize_element(cur_level))
                    else:
                        loc_levels.append(cur_level)

                data["levels"] = loc_levels

            if (
                "codes" in data.keys()
                and len(data["codes"]) > 0
                and isinstance(data["codes"], list)
            ):
                loc_codes = []
                for cur_code in data["codes"]:
                    # Need to test whether we're deserializing or de novo construction
                    if (
                        isinstance(cur_code, dict)
                        and "el" in cur_code.keys()
                        and "eltype" in cur_code.keys()
                    ):
                        loc_codes.append(_deserialize_element(cur_code))
                    else:
                        loc_codes.append(cur_code)

                data["codes"] = loc_codes

        return data


class FIIntervalIndex(FIBaseIndex):
    """Corresponds to pd.IntervalIndex"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.arrays.IntervalArray | np.ndarray
    closed: IntervalClosedType
    name: str | None = None
    dtype: pd.IntervalDtype | str | None

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        return FIIndexType.interval.name

    def get_fi_index_type(self) -> str:
        return FIIndexType.interval

    def get_as_index(self, **kwargs) -> pd.IntervalIndex:
        return pd.IntervalIndex(
            data=self.data,  # type: ignore
            closed=self.closed,
            name=self.name,
            dtype=self.dtype,  # type: ignore
            copy=True,
        )

    @field_serializer("data", when_used="always")
    def serialize_data(self, data: pd.arrays.IntervalArray | np.ndarray):
        return _serialize_element(list(data))

    @field_serializer("dtype", when_used="always")
    def serialize_dtype(self, dtype: Dtype | None):
        return str(dtype)

    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check if data provided is a "true" data array or if it's serialized from before
            # if "data" in data.keys() and len(data["data"]) > 0 and isinstance(data["data"][0], dict) and "eltype" in data["data"][0].keys():
            if (
                "data" in data.keys()
                and isinstance(data["data"], dict)
                and "el" in data["data"].keys()
                and "eltype" in data["data"].keys()
            ):
                data["data"] = _deserialize_element(data["data"])

                # Force IntervalArray
                data["data"] = pd.arrays.IntervalArray(data["data"])

        return data


class FIDatetimeIndex(FIBaseIndex):
    """Corresponds to pd.DatetimeIndex"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: ArrayLike | AnyArrayLike | list | tuple
    freq: _Frequency | None = None
    tz: tzinfo | str | None  # most what it should be from pandas src
    name: str | None = None
    dtype: Dtype | str | None  # Hmmm.

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        return FIIndexType.datetime.name

    def get_fi_index_type(self) -> str:
        return FIIndexType.datetime

    def get_as_index(self, **kwargs) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            data=self.data,
            freq=self.freq,
            tz=self.tz,
            name=self.name,
            dtype=self.dtype,
            copy=True,
        )

    @field_serializer("data", when_used="always")
    def serialize_data(self, data: ArrayLike | AnyArrayLike | list | tuple):
        return _serialize_element(data)

    @field_serializer("freq", when_used="always")
    def serialize_freq(self, freq):
        if self.freq is None:
            return None
        else:
            return freq.freqstr

    @field_serializer("tz", when_used="always")
    def serialize_tz(self, tz):
        if self.tz is None:
            return None
        else:
            return str(self.tz)

    @field_serializer("dtype", when_used="always")
    def serialize_dtype(self, dtype: Dtype | None):
        return str(dtype)

    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check if data provided is a "true" data array or if it's serialized from before
            if (
                "data" in data.keys()
                and isinstance(data["data"], dict)
                and "el" in data["data"].keys()
                and "eltype" in data["data"].keys()
            ):
                data["data"] = _deserialize_element(data["data"])

        return data


class FITimedeltaIndex(FIBaseIndex):
    """Corresponds to pd.TimedeltaIndex"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: ArrayLike | AnyArrayLike | list | tuple
    freq: str | BaseOffset | None = None
    name: str | None = None
    dtype: (
        DtypeObj | np.dtypes.TimeDelta64DType | Literal["<m8[ns]"] | str | None
    )  # Hmmm.

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        return FIIndexType.timedelta.name

    def get_fi_index_type(self) -> str:
        return FIIndexType.timedelta

    def get_as_index(self, **kwargs) -> pd.TimedeltaIndex:
        return pd.TimedeltaIndex(
            data=self.data,  # type: ignore
            freq=self.freq,  # type: ignore
            name=self.name,  # type: ignore
            dtype=self.dtype,  # type: ignore
            copy=True,
        )

    @field_serializer("data", when_used="always")
    def serialize_data(self, data: ArrayLike | AnyArrayLike | list | tuple):
        return _serialize_element(list(data))
        # if isinstance(data, np.ndarray):
        #     loc_list = data.tolist()
        # else:
        #     loc_list = list(data)

        # return list(map(str, loc_list))

    @field_serializer("freq", when_used="always")
    def serialize_freq(self, freq):
        if self.freq is None:
            return None
        else:
            return freq.freqstr

    @field_serializer("dtype", when_used="always")
    def serialize_dtype(self, dtype: Dtype | None):
        return str(dtype)

    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check if data provided is a "true" data array or if it's serialized from before
            # if "data" in data.keys() and len(data["data"]) > 0 and isinstance(data["data"][0], dict) and "eltype" in data["data"][0].keys():
            if (
                "data" in data.keys()
                and isinstance(data["data"], dict)
                and "el" in data["data"].keys()
                and "eltype" in data["data"].keys()
            ):
                data["data"] = _deserialize_element(data["data"])

        return data


class FIPeriodIndex(FIBaseIndex):
    """Corresponds to pd.PeriodIndex"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: ArrayLike | AnyArrayLike | list | tuple
    freq: _Frequency | None = None
    name: str | None = None
    dtype: DtypeObj | pd.PeriodDtype | str | None  # Hmmm.

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        return FIIndexType.period.name

    def get_fi_index_type(self) -> str:
        return FIIndexType.period

    def get_as_index(self, **kwargs) -> pd.PeriodIndex:
        return pd.PeriodIndex(
            data=self.data,
            # freq=self.freq,   -- disabled because it seems to mess things up, info included in dtype and freq is old notation (QE-DEC instead of Q-DEC)
            name=self.name,
            dtype=self.dtype,
            copy=True,
        )

    @field_serializer("data", when_used="always")
    def serialize_data(self, data: ArrayLike | AnyArrayLike | list | tuple):
        return _serialize_element(data)

    @field_serializer("freq", when_used="always")
    def serialize_freq(self, freq):
        if self.freq is None:
            return None
        else:
            return freq.freqstr

    @field_serializer("dtype", when_used="always")
    def serialize_dtype(self, dtype: Dtype | None):
        return str(dtype)

    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check if data provided is a "true" data array or if it's serialized from before
            # if "data" in data.keys() and len(data["data"]) > 0 and isinstance(data["data"][0], dict) and "eltype" in data["data"][0].keys():
            if (
                "data" in data.keys()
                and isinstance(data["data"], dict)
                and "el" in data["data"].keys()
                and "eltype" in data["data"].keys()
            ):
                data["data"] = _deserialize_element(data["data"])

        return data


class FIMetainfo(BaseModel):
    """All the collected metadata we use when saving or loading"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Ironically, this should always just be the filename with no paths
    datafile: Path

    # File format
    file_format: FIFileFormatEnum

    # SHA256 hash
    hash: str | None = None

    # Encoding
    encoding: FIEncoding

    # Serialized dtypes
    serialized_dtypes: dict

    # Index information encoded as a FIIndex object
    index: FIBaseIndex

    # Columns, again, as an FIIndex object
    columns: FIBaseIndex

    @field_serializer("datafile", when_used="always")
    def serialize_datafile(self, datafile: Path):
        return str(datafile)

    @field_serializer("file_format", when_used="always")
    def serialize_file_format(self, file_format: FIFileFormatEnum):
        return file_format.name

    @field_serializer("index", when_used="always")
    def serialize_index(self, index: FIBaseIndex):
        # TODO is this ok if caller does a model_dump_json()?
        return index.model_dump()

    @field_serializer("columns", when_used="always")
    def serialize_columns(self, columns: FIBaseIndex):
        # TODO is this ok if caller does a model_dump_json()?
        return columns.model_dump()

    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Need to ensure the index and columns is created as the correct
            # object type, not just instantiating the base class.
            if "index" in data.keys() and isinstance(data["index"], dict):
                data["index"] = _deserialize_index_dict_to_fi_index(data["index"])

            if "columns" in data.keys() and isinstance(data["columns"], dict):
                data["columns"] = _deserialize_index_dict_to_fi_index(data["columns"])

        return data


def _get_extensions_from_file_format(file_format: FIFileFormatEnum) -> str:
    """Given a FIFileFormatEnum, returns the used filename extension

    Parameters
    ----------
    file_format : FIFileFormatEnum

    Returns
    -------
    str
    """

    if file_format == FIFileFormatEnum.csv:
        return "csv"
    elif file_format == FIFileFormatEnum.parquet:
        return "parq"
    else:
        error_msg = f"Unknown file format"
        logger.error(error_msg)
        raise Exception(error_msg)


def _detect_file_format_from_filename(datafile: Path) -> FIFileFormatEnum:
    """Given a filename, returns the file format as a FIFileFormatEnum

    Parameters
    ----------
    datafile : Path

    Returns
    -------
    FIFileFormatEnum

    Raises
    ------
    Exception
        If not a recognised extension.
    """

    extension = datafile.suffix.lower()

    if extension == "csv":
        return FIFileFormatEnum.csv
    elif extension == "parq" or extension == "parquet":
        return FIFileFormatEnum.parquet
    else:
        error_msg = (
            f"Couldn't find FIFileFormatEnum for {extension}."
            f" Filename={datafile.name}"
        )
        logger.error(error_msg)
        raise Exception(error_msg)


def _check_metafile_name(
    datafile: Path,
    file_format: FIFileFormatEnum,
    metafile: Path | None = None,
) -> Path:
    """Checks or generates the absolute path to the metafile

    Parameters
    ----------
    datafile : Path
        Absolute path to the datafile.
    file_format : FIFileFormatEnum
        Format of datafile.
    metafile : Path | None, optional
        Optional metafile name (will check correctness). If None, will generate and return.

    Returns
    -------
    Path
        Absolute path to metafile.
    """

    # Determine output metafile name and ensure extension is correct
    # TODO think we're misssing something here...
    extension = _get_extensions_from_file_format(file_format)
    if metafile is None:
        loc_metafile = datafile.with_suffix(".yaml")
    else:
        loc_metafile = metafile
        if loc_metafile.suffix != ".yaml":
            error_msg = f"File extension for metadata file must be .yaml. Got {loc_metafile.suffix}"
            logger.error(error_msg)
            raise Exception(error_msg)

    # Check path for datafile and metafile are the same
    if datafile.parent != loc_metafile.parent:
        error_msg = f"Path for datafile and metafile must be the same. datafile={datafile}, loc_metafile={metafile}"
        logger.error(error_msg)
        raise Exception(error_msg)

    return loc_metafile


def _preprocess_inplace(df: pd.DataFrame, encoding: FIEncoding):
    pass


def _preprocess_safe(df: pd.DataFrame, encoding: FIEncoding):

    # Make a copy of the dataframe
    loc_df = copy.deepcopy(df)

    # Modify inplace the copied dataframe
    _preprocess_inplace(loc_df, encoding)

    return loc_df


def _serialize_df_dtypes_to_dict(df: pd.DataFrame):
    """Serializes the dtypes from a data from into a dictionary

    This isn't quite as obvious as it seems because the column label isn't
    necessarily a string. Indeed, it can be a tuple (if a MultiIndex is used for
    columns).
    """

    serialized_dtypes = {}

    col_ctr = 0

    # Loop through and just append to serialized_dtypes unless we get a special
    # type we must manually serialize, e.g. category.
    for col_name, dtype in df.dtypes.to_dict().items():

        # # TODO change JSON encoding to be YAML that encodes in the main write/read
        # loc_col_name = json.dumps(_serialize_element(col_name))

        # if dtype == "category":
        #     dtype_full = df.dtypes[col_name]
        #     assert isinstance(dtype_full, pd.CategoricalDtype)
        #     serialized_dtypes[loc_col_name] = {"dtype_str": str(dtype)}
        #     serialized_dtypes[loc_col_name][
        #         "categories"
        #     ] = dtype_full.categories.to_list()
        #     serialized_dtypes[loc_col_name]["ordered"] = str(dtype_full.ordered)
        # else:
        #     serialized_dtypes[loc_col_name] = {"dtype_str": str(dtype)}


        if dtype == "category":
            dtype_full = df.dtypes[col_name]
            assert isinstance(dtype_full, pd.CategoricalDtype)
            serialized_dtypes[col_name] = {"dtype_str": str(dtype)}
            serialized_dtypes[col_name][
                "categories"
            ] = dtype_full.categories.to_list()
            serialized_dtypes[col_name]["ordered"] = str(dtype_full.ordered)
        else:
            serialized_dtypes[col_name] = {"dtype_str": str(dtype)}


    return serialized_dtypes


def _deserialize_df_types(serialized_dtypes: dict):
    """Deserializes output from `_serialize_df_dtypes_to_dict()`"""

    deserialized_dtypes = {}
    for col in serialized_dtypes:
        # Get col name
        # loc_col_name = _deserialize_element(json.loads(col))
        loc_col_name = col

        # Get string representation of dtype
        dtype_str = serialized_dtypes[col].get("dtype_str", None)
        if dtype_str is None:
            error_msg = (
                f"Got column in serialized dtypes without a dtype_str field. col={col}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        deserialized_dtypes[loc_col_name] = {}
        deserialized_dtypes[loc_col_name]["dtype_str"] = dtype_str

        # Check if it's a categorydtype
        if dtype_str == "category":
            deserialized_dtypes[loc_col_name]["categories"] = serialized_dtypes[col][
                "categories"
            ]
            deserialized_dtypes[loc_col_name]["ordered"] = serialized_dtypes[col][
                "ordered"
            ]

    return deserialized_dtypes


def _apply_serialized_dtypes(df: pd.DataFrame, serialized_dtypes: dict):
    """Apply dtypes to dataframe _inplace_

    Parameters
    ----------
    df : pd.DataFrame
        INPLACE dataframe to apply dtypes to.
    serialized_dtypes : dict
        The serialized types obtained from `_serialize_df_dtypes_to_dict()`.
    """

    deserialized_dtypes = _deserialize_df_types(serialized_dtypes)

    for col, dtype_info in deserialized_dtypes.items():
        # Set the dtype for the column
        if dtype_info["dtype_str"] == "category":
            # Construct the dtype
            cat_type = pd.CategoricalDtype(
                categories=dtype_info["categories"],
                ordered=bool(dtype_info["ordered"]),
            )
            df[col] = df[col].astype(cat_type)
        else:
            df[col] = df[col].astype(dtype_info["dtype_str"])

    return None


def _serialize_index_to_metainfo_index(idx: pd.Index) -> FIBaseIndex:
    """Serializes a Pandas index into one of our FI*Index classes"""

    if not isinstance(idx, pd.Index):
        error_msg = f"Must pass a Pandas Index. Got a {type(idx)}"
        logger.error(error_msg)
        raise Exception(error_msg)

    if isinstance(idx, pd.RangeIndex):
        assert isinstance(idx.start, int)
        assert isinstance(idx.stop, int)
        assert isinstance(idx.step, int)
        return FIRangeIndex(
            start=int(idx.start),
            stop=int(idx.stop),
            step=int(idx.step),
            name=idx.name,
            dtype=idx.dtype,
        )
    elif isinstance(idx, pd.CategoricalIndex):
        return FICategoricalIndex(
            data=idx.array.to_numpy(),  # TODO check me
            categories=idx.categories.values,
            ordered=idx.ordered,  # type: ignore
            name=idx.name,
            dtype=idx.dtype,
        )

    elif isinstance(idx, pd.MultiIndex):
        return FIMultiIndex(
            levels=idx.levels,
            codes=idx.codes,
            sortorder=idx.sortorder,  # type: ignore
            names=idx.names,
            dtypes=idx.dtypes,
        )

    elif isinstance(idx, pd.IntervalIndex):
        return FIIntervalIndex(
            data=idx.array,  # type: ignore
            closed=idx.closed,
            name=idx.name,
            dtype=idx.dtype,  # type: ignore
        )
    elif isinstance(idx, pd.DatetimeIndex):
        return FIDatetimeIndex(
            data=idx.array,
            freq=idx.freq,
            tz=idx.tz,
            name=idx.name,
            dtype=idx.dtype,
        )
    elif isinstance(idx, pd.TimedeltaIndex):
        return FITimedeltaIndex(
            data=idx.array,
            freq=idx.freq,
            name=idx.name,
            dtype=idx.dtype,
        )
    elif isinstance(idx, pd.PeriodIndex):
        return FIPeriodIndex(
            data=idx.array,
            freq=idx.freq,
            name=idx.name,
            dtype=idx.dtype,
        )

    # This should be at the end as it's the least specific, i.e. if you put if
    # higher in the list then it'd catch the other index types.
    elif isinstance(idx, pd.Index):
        return FIIndex(
            data=idx.array,
            name=idx.name,
            dtype=idx.dtype,
        )

    else:
        error_msg = f"Unrecognised index type: {type(idx)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def _deserialize_index_dict_to_fi_index(index: dict) -> FIBaseIndex:
    """Deserializes an index stored as a dictionary (from YAML file) into on of our FI*Indexes"""

    index_type = index.get("index_type", None)
    if index_type is None or index_type == "":
        error_msg = f"index_type cannot be empty or None when deserializing."
        logger.error(error_msg)
        raise Exception(error_msg)

    if index_type == FIIndexType.idx:
        return FIIndex(**index)
    elif index_type == FIIndexType.range:
        return FIRangeIndex(**index)
    elif index_type == FIIndexType.categorical:
        return FICategoricalIndex(**index)
    elif index_type == FIIndexType.multi:
        return FIMultiIndex(**index)
    elif index_type == FIIndexType.interval:
        return FIIntervalIndex(**index)
    elif index_type == FIIndexType.datetime:
        return FIDatetimeIndex(**index)
    elif index_type == FIIndexType.timedelta:
        return FITimedeltaIndex(**index)
    elif index_type == FIIndexType.period:
        return FIPeriodIndex(**index)
    else:
        error_msg = f"index_type not recognised. index_type={index_type}"
        logger.error(error_msg)
        raise Exception(error_msg)


def _compile_metainfo(
    datafile: Path,
    file_format: FIFileFormatEnum,
    hash: str,
    encoding: FIEncoding,
    df: pd.DataFrame,
) -> FIMetainfo:
    """Creates an FIMetainfo object from supplied metainfo

    Parameters
    ----------
    datafile : Path
        The associated datafile TODO check if this should be with or without path
    file_format : FIFileFormatEnum
        The datafile format.
    hash : str
        Hash of the datafile.
    encoding : FIEncoding
        Encoding options.
    df : pd.DataFrame
        The actual dataframe (we need to record information about indexes and dtypes).

    Returns
    -------
    FIMetainfo
    """

    # Get dtypes as a dictionary
    serialized_dtypes = _serialize_df_dtypes_to_dict(df)

    # Get index as an FIIndex object
    index = _serialize_index_to_metainfo_index(df.index)

    # Get columns as an FIIndex object
    columns = _serialize_index_to_metainfo_index(df.columns)

    # Now shove it all into a FIMetainfo object...
    metainfo = FIMetainfo(
        datafile=Path(datafile.name),
        file_format=file_format,
        hash=hash,
        encoding=encoding,
        serialized_dtypes=serialized_dtypes,
        index=index,
        columns=columns,
    )

    return metainfo


def _write_to_csv(df: pd.DataFrame, datafile: Path, encoding: FIEncoding):
    """Write to a CSV datafile

    This is mainly about applying relevant options for the write.

    Parameters
    ----------
    df : pd.DataFrame
    datafile : Path
    encoding : FIEncoding
    """

    df.to_csv(
        datafile,
        header=True,
        index=True,
        doublequote=encoding.csv.doublequote,
        sep=encoding.csv.sep,
        na_rep=encoding.csv.na_rep,
        quoting=encoding.csv.quoting,  # type: ignore
    )


def _read_from_csv(
    input_datafile: Path,
    encoding: FIEncoding,
    num_index_cols: int = 1,
    num_index_rows: int = 1,
) -> pd.DataFrame:
    """Read from CSV file using supplied options"""

    if num_index_cols == 1:
        index_col = [0]
    else:
        index_col = list(range(0, num_index_cols))

    if num_index_rows == 1:
        index_row = [0]
    else:
        index_row = list(range(0, num_index_rows))

    df = pd.read_csv(
        input_datafile,
        header=index_row,
        index_col=index_col,
        float_precision=encoding.csv.float_precision,
        doublequote=encoding.csv.doublequote,
        sep=encoding.csv.sep,
        dtype=None,
        keep_default_na=encoding.csv.keep_default_na,
        na_values=encoding.csv.csv_allowed_na,
    )

    return df


def _write_to_parquet(df: pd.DataFrame, datafile: Path, encoding: FIEncoding):
    """Write to Parquet datafile using supplied options

    Parameters
    ----------
    df : pd.DataFrame
    datafile : Path
    encoding : FIEncoding
    """
    df.to_parquet(datafile, engine="pyarrow", index=True)


def _read_from_parquet(input_datafile: Path, encoding: FIEncoding) -> pd.DataFrame:
    df = pd.read_parquet(input_datafile, engine="pyarrow")
    return df


def _write_metafile(datafile: Path, metafile: Path, metainfo: FIMetainfo):
    """Write a FIMetainfo object to a YAML file

    Parameters
    ----------
    datafile : Path
        The datafile the metainfo file describes.
    metafile : Path
        The YAML file we're dumping this metainfo into.
    metainfo : FIMetainfo
        The metainfo.
    """

    # Get a dict from the FIMetainfo model
    metainfo_dict = metainfo.model_dump()

    # Get YAML string
    yaml_output = yaml.dump(metainfo_dict)

    # Write to the YAML file
    with open(metafile, "w") as h_targetfile:
        h_targetfile.write(f"# Metadata for {str(datafile)}\n")
        h_targetfile.write("---\n\n")

        # Write out the rest of the file now
        h_targetfile.write(yaml_output)


def _read_metafile(metafile: Path) -> FIMetainfo:
    """Reads in metainfo from file"""

    # Read metainfo from file
    with open(metafile, "r") as h_metafile:
        metainfo_dict = yaml.load(h_metafile, Loader=Loader)

    # Convert into an FIMetainfo model
    metainfo = FIMetainfo(**metainfo_dict)

    return metainfo


# need to specify output format
def write_df_to_fi_generic(
    df: pd.DataFrame,
    datafile: Path,
    metafile: Path | None = None,
    file_format: FIFileFormatEnum | None = None,
    encoding: FIEncoding | None = None,
    preprocess_inplace=True,
) -> Path:
    """Writes a dataframe to file

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to save.
    datafile : Path
        The datafile to save the dataframe to.
    metafile : Path | None, optional
        Metafile name. If not supplied will be determined automatically.
    file_format : FIFileFormatEnum | None, optional
        The file format. If not supplied will be determined automatically.
    encoding : FIEncoding | None, optional
        Datafile encoding options.
    preprocess_inplace : bool, optional

    Returns
    -------
    Path
        A Path object with the metainfo filename in it.

    """

    # Determine output format
    if file_format is None:
        loc_file_format = _detect_file_format_from_filename(datafile)
    else:
        loc_file_format = FIFileFormatEnum(file_format)

    # Determine metafile name
    loc_metafile = _check_metafile_name(datafile, loc_file_format, metafile)

    # If we've got encoding parameters, use them; otherwise use defaults
    if encoding is None:
        encoding = FIEncoding()

    # Preprocess
    if preprocess_inplace:
        _preprocess_inplace(df, encoding)
        loc_df = df
    else:
        loc_df = _preprocess_safe(df, encoding)

    # Write to the data file
    if loc_file_format == FIFileFormatEnum.csv:
        _write_to_csv(loc_df, datafile, encoding)
    elif loc_file_format == FIFileFormatEnum.parquet:
        _write_to_parquet(loc_df, datafile, encoding)
    else:
        error_msg = f"Output format not supported. This shouldn't happen."
        logger.error(error_msg)
        raise Exception(error_msg)

    # Calculate the file's hash
    with open(datafile, "rb") as h_datafile:
        digest = hashlib.file_digest(h_datafile, "sha256")
    hash = digest.hexdigest()

    # Compile all the metainfo into a dictionary TODO need other metainfo here!
    metainfo = _compile_metainfo(
        datafile=datafile,
        file_format=loc_file_format,
        hash=hash,
        encoding=encoding,
        df=loc_df,
    )

    # Write metafile
    _write_metafile(datafile, loc_metafile, metainfo)

    return loc_metafile


def read_fi_to_df_generic(
    metafile: Path, strict_hash_check: bool = True
) -> tuple[pd.DataFrame, FIMetainfo]:
    """Load a dataframe from file

    Supply the metainfo filename, not the datafilename.

    Parameters
    ----------
    metafile : Path
        The YAML file that is associated with the datafile.
    strict_hash_check : bool, optional
        Whether we raise an exception if the hash is wrong.

    Returns
    -------
    tuple[pd.DataFrame, FIMetainfo]:
        A tuple with the dataframe and the metainfo object.
    """

    # Load metainfo
    metainfo = _read_metafile(metafile)

    # Check datafile's hash
    datafile_abs = Path(metafile.parent / metainfo.datafile).resolve()
    with open(datafile_abs, "rb") as h_datafile:
        digest = hashlib.file_digest(h_datafile, "sha256")
    hash = digest.hexdigest()
    if hash != metainfo.hash:
        error_msg = f"Hash comparison failed. metainfo.hash={metainfo.hash}, calcualted hash={hash}."
        if strict_hash_check:
            logger.error(error_msg)
            raise Exception(error_msg)
        else:
            logger.warning(error_msg)

    # Need to know number of columns
    if isinstance(metainfo.index, FIMultiIndex):
        num_index_cols = len(metainfo.index.levels)
    else:
        num_index_cols = 1

    if isinstance(metainfo.columns, FIMultiIndex):
        num_index_rows = len(metainfo.columns.levels)
    else:
        num_index_rows = 1

    # TODO need to check index rows and cols the right way around

    # Load the data
    if metainfo.file_format == FIFileFormatEnum.csv:
        df = _read_from_csv(
            datafile_abs,
            metainfo.encoding,
            num_index_cols=num_index_cols,
            num_index_rows=num_index_rows,
        )
    elif metainfo.file_format == FIFileFormatEnum.parquet:
        df = _read_from_parquet(datafile_abs, metainfo.encoding)
    else:
        error_msg = f"Input format not supported. This shouldn't happen."
        logger.error(error_msg)
        raise Exception(error_msg)

    # Apply index and columns
    df.index = metainfo.index.get_as_index()
    df.columns = metainfo.columns.get_as_index()

    # Apply dtypes
    _apply_serialized_dtypes(df, metainfo.serialized_dtypes)

    return (df, metainfo)


def _generate_example_indices():

    fi_index_1 = pd.Index(
        [
            "a",
            "b",
            "c",
            1,
            2,
            3,
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02 10:11:12"),
            "an arbitrary string",
        ]
    )
    fi_index_2 = pd.Index(
        [
            "a",
            "b",
            1,
            2,
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02 10:11:12"),
            "an arbitrary string",
        ],
        name="index2",
    )

    fi_rangeindex_1 = pd.RangeIndex(start=-1000, stop=1000, step=10)
    fi_rangeindex_2 = pd.RangeIndex(
        start=-10000, stop=-5123, step=11, name="rangeindex2"
    )
    fi_rangeindex_3 = pd.RangeIndex(start=9, stop=-21, step=-3, name="rangeindex3")

    fi_categoricalindex_1 = pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"])
    fi_categoricalindex_2 = pd.CategoricalIndex(
        ["a", "b", "c", "a", "b", "c"], ordered=True, name="categoricalindex2"
    )
    fi_categoricalindex_3 = pd.CategoricalIndex(
        [1, 1, 2, 1, 1, 3, 1, 1, 5, 6, 7, 1, 2, 3, 4, 5],
        ordered=True,
        name="categoricalindex3",
    )
    fi_categoricalindex_4 = pd.CategoricalIndex([0.1, 0.2, 0.3, 0.5, 0.1], ordered=True)

    mi_arrays_1 = [[1, 2, 3, 4], ["a", "b", "c"], ["one", "two", "three"]]
    mi_arrays_2 = [
        [1, 2, 3, 4],
        [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
    ]
    fi_multiindex_1 = pd.MultiIndex.from_product(mi_arrays_1)
    fi_multiindex_2 = pd.MultiIndex.from_product(
        mi_arrays_1, names=["num", "letters", "other"]
    )
    fi_multiindex_3 = pd.MultiIndex.from_product(mi_arrays_2, names=["number", "date"])

    fi_intervalindex_1 = pd.interval_range(start=0, end=10, closed="left")
    fi_intervalindex_2 = pd.interval_range(
        start=0, end=50, freq=2, closed="both", name="intervalindex2"
    )
    fi_intervalindex_3 = pd.interval_range(
        start=pd.Timestamp("2024-02-01"),
        end=pd.Timestamp("2024-04-30"),
        freq="2D",
        closed="neither",
        name="intervalindex3",
    )
    fi_intervalindex_4 = pd.interval_range(start=-3.0, end=11.0, freq=0.5, closed="neither", name="intervalindex4")  # type: ignore

    fi_datetimeindex_1 = pd.DatetimeIndex(
        data=["2024-01-01 10:00:00", "2024-01-02 10:00:00", "2024-01-03 10:00:00"],
        freq="D",
        tz="EST",
    )
    fi_datetimeindex_2 = pd.date_range(
        start=pd.to_datetime("1/1/2018").tz_localize("Europe/Berlin"),
        end=pd.to_datetime("1/08/2018").tz_localize("Europe/Berlin"),
    )
    fi_datetimeindex_3 = pd.DatetimeIndex(
        data=["2024-01-01 10:05:00", "2024-01-02 11:07:00", "2024-01-03 09:00:00"],
        tz="UTC",
    )

    fi_periodindex_1 = pd.PeriodIndex.from_fields(year=[2000, 2002, 2004], quarter=[1, 3, 2])  # type: ignore
    fi_periodindex_2 = pd.period_range(start="2017-01-01", end="2018-01-01", freq="M")

    return {
        "fi_index_1": fi_index_1,
        "fi_index_2": fi_index_2,
        "fi_rangeindex_1": fi_rangeindex_1,
        "fi_rangeindex_2": fi_rangeindex_2,
        "fi_rangeindex_3": fi_rangeindex_3,
        "fi_categoricalindex_1": fi_categoricalindex_1,
        "fi_categoricalindex_2": fi_categoricalindex_2,
        "fi_categoricalindex_3": fi_categoricalindex_3,
        "fi_categoricalindex_4": fi_categoricalindex_4,
        "fi_multiindex_1": fi_multiindex_1,
        "fi_multiindex_2": fi_multiindex_2,
        "fi_multiindex_3": fi_multiindex_3,
        "fi_intervalindex_1": fi_intervalindex_1,
        "fi_intervalindex_2": fi_intervalindex_2,
        "fi_intervalindex_3": fi_intervalindex_3,
        "fi_intervalindex_4": fi_intervalindex_4,
        "fi_datetimeindex_1": fi_datetimeindex_1,
        "fi_datetimeindex_2": fi_datetimeindex_2,
        "fi_datetimeindex_3": fi_datetimeindex_3,
        "fi_periodindex_1": fi_periodindex_1,
        "fi_periodindex_2": fi_periodindex_2,
    }


def _generate_dfs_from_indices(test_indices):

    dfs = {}
    for k, v in test_indices.items():
        idx_len = len(v)

        # Create a dataframe with index used both as row index and column index
        df = pd.DataFrame(np.random.randn(idx_len, idx_len), index=v, columns=v)

        dfs[k] = df

    return dfs


def generate_test_df_static_1():

    # See
    # https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
    # https://pandas.pydata.org/docs/reference/arrays.html#api-arrays-datetime

    dates = pd.date_range("1/1/2000", periods=5, name="Date")
    df = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=["A", "B", "C"])

    # NumPy data types
    # ----------

    df["F_np_int8"] = pd.array([1, 0, -50, -127, 127], dtype="int8")
    df["F_np_int16"] = pd.array([1, 0, -50, -32000, 32000], dtype="int16")
    df["F_np_int32"] = pd.array([1, 0, -50, -2000000000, 2000000000], dtype="int32")
    df["F_np_int64"] = pd.array([1, 0, -50, -4000000000, 4000000000], dtype="int64")
    df["F_np_longlong"] = pd.array(
        [1, 0, -50, -4000000000, 4000000000], dtype="longlong"
    )
    # df["F_np_timedelta64"]  = pd.array([1, 0, -50, -4000000000, 4000000000], dtype=np.timedelta64)

    df["F_np_uint8"] = pd.array([1, 0, 50, 240, 127], dtype="uint8")
    df["F_np_uint16"] = pd.array([1, 0, 50, 3200, 32000], dtype="uint16")
    df["F_np_uint32"] = pd.array([1, 0, 50, 2000000000, 2000000000], dtype="uint32")
    df["F_np_uint64"] = pd.array([1, 0, 50, 4000000000, 4000000000], dtype="uint64")
    df["F_np_ulonglong"] = pd.array(
        [1, 0, 50, 4000000000, 4000000000], dtype="ulonglong"
    )

    df["F_np_float16"] = pd.array([1.0, -2.0, np.pi, np.NaN, 5.0], dtype="float16")
    df["F_np_float32"] = pd.array([1.0, -2.0, np.pi, np.NaN, 5.0], dtype="float32")
    df["F_np_float64"] = pd.array([1.0, -2.0, np.pi, np.NaN, 5.0], dtype="float64")

    df["F_np_complex64"] = pd.array(
        [1.0 + 1.0j, -2.0 - 1j, np.pi * 1j, np.NaN, 5.0 - 800j], dtype="complex64"
    )
    df["F_np_complex128"] = pd.array(
        [1.0 + 1.0j, -2.0 - 1j, np.pi * 1j, np.NaN, 5.0 - 800j], dtype="complex128"
    )
    df["F_np_clongdouble"] = pd.array(
        [1.0 + 1.0j, -2.0 - 1j, np.pi * 1j, np.NaN, 5.0 - 800j], dtype="clongdouble"
    )

    # other
    df["F_np_bool"] = pd.array([True, False, True, True, False], dtype="bool_")
    df["F_np_datetime64"] = pd.array(
        [
            np.datetime64("2010-01-31T10:23:01"),
            np.datetime64("1990"),
            np.datetime64("2025-01-01T00:00"),
            1,
            np.datetime64("2010-12-31T15:00"),
        ],
        dtype="datetime64[ns]",
    )

    # Pandas extended data types
    # ----------

    # Careful with syntax! See, https://github.com/pandas-dev/pandas/issues/57644
    # df["F_pd_DatetimeTZDtype"] = pd.array(
    #     ["2010/01/31 10:23:01", "1990", "2025/01/01 00:00+1", 1, None],
    #     dtype=pd.DatetimeTZDtype(tz=ZoneInfo("Europe/Paris")),
    # 
    df["F_pd_DatetimeTZDtype"] = pd.array(
        ["2010/01/31 10:23:01", "1990", "2025/01/01 00:00+1", 1, None],
        dtype="datetime64[ns, Europe/Paris]",
    )


    # Missing: Timedeltas
    # Missing: PeriodDtype
    # Missing: IntervalDtype
    df["F_pd_Int64Dtype"] = pd.array(
        [1, None, -50, -1000000000, 1000000000], dtype=pd.Int64Dtype()
    )
    df["F_pd_Float64Dtype"] = pd.array(
        [1.0, -2.0, np.pi, None, 5.0], dtype=pd.Float64Dtype()
    )
    df["F_pd_CategoricalDtype"] = pd.Categorical(
        ["a", "b", "c", "a", "a"], categories=["a", "b", "c"], ordered=True
    )
    df["F_pd_StringDtype"] = pd.array(
        ["this", "col", "is string", "based", "!"], dtype=pd.StringDtype()
    )
    # Missing: Sparse
    df["F_pd_BooleanDtype"] = pd.array(
        [True, False, None, True, False], dtype=pd.BooleanDtype()
    )
    # Missing: ArrowDtype

    return df


def test_save_load_indices(test_path: Path):

    if not test_path.exists():
        test_path.mkdir(parents=True)

    # Get test indices
    test_indices = generate_test_indices()

    # Get randoms dfs using these indices
    dfs = generate_test_dfs_from_indices(test_indices)

    # For each, save and reload, then compare
    for k, df in dfs.items():

        # if re.match(".*multi.*", k, re.IGNORECASE):
        #     print("Ignoring {k}")
        #     continue

        print(f"Testing {k}")

        # Save
        datafile_abs = test_path / f"test_{k}.csv"
        metafile_abs = test_path / f"test_{k}.yaml"
        metafile = write_df_to_fi_generic(
            df, datafile_abs, metafile_abs, FIFileFormatEnum.csv
        )

        # Reload
        (df_reload, metainfo_reload) = read_fi_to_df_generic(metafile)

        # Compare
        pd.testing.assert_frame_equal(df, df_reload)
