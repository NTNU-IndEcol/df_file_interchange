"""
Generic version of the File Interchange code.

Warning: We set CoW semantics (will be in Pandas 3.0 anyway)!

Note to self:

* Was going to allow user to derive from a FICustomInfo class to make
  serialization possible but this gets difficult:
  https://blog.devgenius.io/deserialize-child-classes-with-pydantic-that-gonna-work-784230e1cf83
  (there's no obvious way to allow users to define the union type) So, we'd be
  left with having to store the class name and we're not guaranteed that'll
  exist again on deserialize. Basically, not worth the effort so will use a
  plain dict.
* pandas.Float64Dtype converts np.NaN and np.Inf to <NA>.
* Have two deserialize for dtypes: one that's used for applying dtypes using
  `.astype()` used after all reads; and, the other, used only for CSV when
  calling `read_csv()`.

Pandas issues:

* Generally unable to use to_json() with the dtype: it's broken in several places.
* There's issues with ordering when using CategoricalIndex in a Multiindex. See
  https://stackoverflow.com/questions/71837659/trying-to-sort-multiindex-index-using-categorical-index
  and https://github.com/pandas-dev/pandas/issues/47607

Pandas sorted issues:

* assert_frame_equal() fails when it shouldn't. I'm really tired of this
  nonsense now. See https://github.com/pandas-dev/pandas/issues/57644 for
  description/resolution of sorts.

Parquet issues / properties:

* Pyarrow doesn't permit different types within a column, which also applies to
  the row index (it fails). It complains if one tries similar with column index.

Discussion:

Code must be able to write out and read in exactly the same dataframe. This is a
a fair requirement but turns out to be more tricky with Pandas than it should
be. So instead of an elegant solution, have had to manually write code to
properly serialise column dtype information.



"""

import copy
import csv
import sys
from datetime import tzinfo
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeAlias, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pandas._libs.tslibs import BaseOffset
from pandas._testing import assert_frame_equal

# Pylance complains about Frequency, presumably because it uses a ForwardRef.
# It's not striclty wrong but will redefine locally for now: see _Frequency
# later on. Can probably fix this with TYPE_CHECKING stuff but will need to read
# up on it.
from pandas._typing import AnyArrayLike, ArrayLike, Dtype, DtypeObj, IntervalClosedType

# DO NOT try to remove these. It's required by Pydantic to resolve forward
# references, I think. Anyway, it raises an exception without these.
from pandas import Index, Series  # noqa: F401
from pandas.api.extensions import ExtensionArray, ExtensionDtype  # noqa: F401

# Pydantic imports
from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_serializer,
    model_validator,
    ValidationInfo,
    field_validator,
    SerializeAsAny,
)

# Import common functions
from ..util.common import str_n, safe_str_output, hash_file

# Import custom info stuff
from ..ci.base import FIBaseCustomInfo

# Import custom info stuff. DO NOT remove these. They're required when
# dynamically checking inheritance of the structured info/unit classes when
# instantiating.
from ..ci.extra.base import FIBaseExtraInfo  # noqa: F401
from ..ci.extra.std_extra import FIStdExtraInfo  # noqa: F401
from ..ci.unit.base import FIBaseUnit  # noqa: F401
from ..ci.unit.currency import FICurrencyUnit  # noqa: F401
from ..ci.unit.population import FIPopulationUnit  # noqa: F401
from ..ci.structured import FIStructuredCustomInfo  # noqa: F401


# Setup YAML
try:
    from yaml import CDumper as Dumper  # noqa: F401
    from yaml import CLoader as Loader  # noqa: F401
except ImportError:
    from yaml import Loader  # noqa: F401

# Set CoW semantics
pd.set_option("mode.copy_on_write", True)
pd.options.mode.copy_on_write = True

# TODO this isn't nice. Pylance complains about
_Frequency = Union[str, BaseOffset]

# Our "anything like a list"
TArrayThing: TypeAlias = list | tuple | np.ndarray


class InvalidValueForFieldError(Exception):
    """Used to indicate a value has been passed to a field of unsuitable type, e.g. passing int to a float dtype field"""

    pass


def chk_strict_frames_eq_ignore_nan(df1: pd.DataFrame, df2: pd.DataFrame):
    """Check whether two dataframes are equal, ignoring NaNs

    This may be expensive since we have to make a copy of the dataframes to
    avoid mangling the originals. Raises exception if dfs are unequal.

    Parameters
    ----------
    df1 : pd.DataFrame
    df2 : pd.DataFrame

    Returns
    -------
    bool
        Always True.
    """

    const_float = np.pi

    # Copy the dataframes because we do not want to modify the originals
    loc_df1 = df1.copy()
    loc_df2 = df2.copy()

    # Iterate through the columns of each df, and if the column dtype is float then we replace NaNs with a finite value
    # We don't seem to need to bother to do this for np.complex types, so this seems ok...
    col_list1 = []
    col_list2 = []
    for col in loc_df1:
        if loc_df1[col].dtype in ["float16", "float32", "float64"]:
            col_list1.append(col)
    for col in loc_df2:
        if loc_df2[col].dtype in ["float16", "float32", "float64"]:
            col_list2.append(col)

    d_col_list1 = {col: {np.nan: const_float} for col in col_list1}
    d_col_list2 = {col: {np.nan: const_float} for col in col_list2}
    loc_df1.replace(to_replace=d_col_list1, inplace=True)
    loc_df2.replace(to_replace=d_col_list2, inplace=True)

    # Finallly, we can do the test free from NaN != NaN issues.
    assert_frame_equal(
        loc_df1,
        loc_df2,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_categorical=True,
        check_frame_type=True,
        check_names=True,
        check_exact=True,
        check_freq=True,
        # check_flag=True,
    )

    return True


def _check_valid_scalar_generic_cast(val, dtype):
    """Checks whether casting `val` to type `dtype` is valid

    Use this for Python types. `_check_valid_scalar_np_cast()` is for NumPy
    types.

    Not sure if this is strictly necessary since it'd likely be caught on an
    explicit case anyway but, meh.

    Parameters
    ----------
    val : Any
    dtype : type

    Raises
    ------
    InvalidValueForFieldError
        If not a valid cast.
    """

    if dtype(val) != val:
        error_msg = f"Value is not of target type. val={safe_str_output(val)}, target dtype={safe_str_output(dtype)}"
        logger.error(error_msg)
        raise InvalidValueForFieldError(error_msg)


def _check_valid_scalar_np_cast(val, dtype):
    """Checks whether casting `val` to type `dtype` is valid

    Use this for NumPy types. `_check_valid_scalar_generic_cast()` is for NumPy
    types.

    Not sure if this is strictly necessary since it'd likely be caught on an
    explicit case anyway but, meh.

    Parameters
    ----------
    val : Any
    dtype : numpy type

    Raises
    ------
    InvalidValueForFieldError
        If not a valid cast.
    """

    if not np.can_cast(val, dtype):
        error_msg = f"Value is not of target type. val={safe_str_output(val)}, target dtype={safe_str_output(dtype)}"
        logger.error(error_msg)
        raise InvalidValueForFieldError(error_msg)


def _serialize_element(el, b_only_known_types: bool = True) -> dict:
    """Serialize something

    This is used primarily to encode indexes.

    Parameters
    ----------
    el : list, tuple, array, int, str, float, etc.
    b_only_known_types : bool
        Default True. If True will raise exception if we encounter a type we
        don't know about.

    Returns
    -------
    dict
        Field `el` contains the serialized content. Field `eltype` describes how
        it was serialized.
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
        # This isn't ideal since it's replicating functionality that's in FIIndex
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
        loc_el = int(el)  # See #6
        loc_type = "np.int8"
    elif type(el) == np.int16:
        loc_el = int(el)  # See #6
        loc_type = "np.int16"
    elif type(el) == np.int32:
        loc_el = int(el)  # See #6
        loc_type = "np.int32"
    elif type(el) == np.int64:
        loc_el = int(el)  # See #6
        loc_type = "np.int64"
    elif type(el) == np.longlong:
        loc_el = int(el)  # See #6
        loc_type = "np.longlong"
    elif type(el) == np.uint8:
        loc_el = int(el)  # See #6
        loc_type = "np.uint8"
    elif type(el) == np.uint16:
        loc_el = int(el)  # See #6
        loc_type = "np.uint16"
    elif type(el) == np.uint32:
        loc_el = int(el)  # See #6
        loc_type = "np.uint32"
    elif type(el) == np.uint64:
        loc_el = int(el)  # See #6
        loc_type = "np.uint64"
    elif type(el) == np.ulonglong:
        loc_el = int(el)  # See #6
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
    elif type(el) == np.complex128:
        loc_el = complex(el)
        loc_type = "np.complex128"
    elif type(el) == np.complex256:
        loc_el = complex(el)
        loc_type = "np.complex256"
    elif type(el) == np.clongdouble:
        loc_el = complex(el)
        loc_type = "np.clongdouble"
    else:
        if b_only_known_types:
            error_msg = f"We only serialize types we know. Got type(el)={type(el)}"
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
        else:
            warning_msg = (
                f"Got unexpected type on serialize element. type(el)={type(el)}"
            )
            logger.warning(warning_msg)
            loc_el = el
            loc_type = ""

    return {"el": loc_el, "eltype": loc_type}


def _deserialize_element(
    serialized_element: dict,
    b_chk_correctness: bool = True,
    b_only_known_types: bool = True,
):
    """Deserialize a dict created by `_serialize_element()`

    Parameters
    ----------
    serialized_element : dict
        The serialized element we want to deserialize.
    b_chk_correctness : bool
        Default True. Does some checking of type casts. You should probably keep
        this as True.
    b_only_known_types : bool
        Default True. If True will raise exception if we encounter a type we
        don't know about.
    """

    # TODO recursion protection

    if (
        "el" not in serialized_element.keys()
        or "eltype" not in serialized_element.keys()
    ):
        error_msg = (
            "Cannot unserialize: both 'el' and 'eltype' fields must be specified."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    el = serialized_element["el"]
    eltype = serialized_element["eltype"]

    # Basic sanity checks. This shouldn't happen in normal operation but have
    # done stupid stuff when writing tests, debugging, etc.
    if not isinstance(eltype, str):
        error_msg = f"Need eltype to be a string. Got type(eltype)={type(eltype)}, eltype={safe_str_output(eltype)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Do the deserialize
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
        return pd.arrays.DatetimeArray._from_sequence(  # type: ignore  (look, this isn't my fault, see example in https://pandas.pydata.org/docs/reference/api/pandas.arrays.DatetimeArray.html )
            _deserialize_list_with_types(el["elements"]), dtype=el["dtype"], copy=True
        )  # type: ignore
    elif eltype == "pd.arrays.PeriodArray":
        return pd.arrays.PeriodArray._from_sequence(  # type: ignore
            _deserialize_list_with_types(el["elements"]), dtype=el["dtype"], copy=True
        )  # type: ignore
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
        if b_chk_correctness:
            _check_valid_scalar_generic_cast(el, str)
        return str(el)
    elif eltype == "int":
        if b_chk_correctness:
            _check_valid_scalar_generic_cast(el, int)
        return int(el)
    elif eltype == "float":
        if b_chk_correctness:
            _check_valid_scalar_generic_cast(el, float)
        return float(el)
    elif eltype == "np.int8":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.int8)
        return np.int8(el)
    elif eltype == "np.int16":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.int16)
        return np.int16(el)
    elif eltype == "np.int32":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.int32)
        return np.int32(el)
    elif eltype == "np.int64":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.int64)
        return np.int64(el)
    elif eltype == "np.longlong":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.longlong)
        return np.longlong(el)
    elif eltype == "np.uint8":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.uint8)
        return np.uint8(el)
    elif eltype == "np.uint16":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.uint16)
        return np.uint16(el)
    elif eltype == "np.uint32":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.uint32)
        return np.uint32(el)
    elif eltype == "np.uint64":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.uint64)
        return np.uint64(el)
    elif eltype == "np.ulonglong":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.ulonglong)
        return np.ulonglong(el)
    elif eltype == "np.float16":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.float16)
        return np.float16(el)
    elif eltype == "np.float32":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.float32)
        return np.float32(el)
    elif eltype == "np.float64":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.float64)
        return np.float64(el)
    elif eltype == "np.complex64":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.complex64)
        return np.complex64(el)
    elif eltype == "np.complex128":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.complex128)
        return np.complex128(el)
    elif eltype == "np.complex256":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.complex256)
        return np.complex256(el)
    elif eltype == "np.clongdouble":
        if b_chk_correctness:
            _check_valid_scalar_np_cast(el, np.clongdouble)
        return np.clongdouble(el)
    else:
        if b_only_known_types:
            error_msg = f"We only deserialize types we know. Got eltype={safe_str_output(eltype)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        else:
            warning_msg = f"In deserialize element, got unknown type. Got eltype={safe_str_output(eltype)}"
            logger.warning(warning_msg)
            return el


def _serialize_list_with_types(data: list | tuple | np.ndarray) -> list:
    """Serialize a list (don't call this directly, use `_serialize_element()` instead)

    Serializes list/tuple/np.ndarary elementwise.

    Parameters
    ----------
    data : list | tuple | np.ndarray

    Returns
    -------
    list
    """

    loc_data = []
    for item in data:
        loc_data.append(_serialize_element(item))

    return loc_data


def _deserialize_list_with_types(serialized_data: list[dict]) -> list:
    """Counterpart to `_serialize_list_with_types()` (again, don't call this directly)

    Parameters
    ----------
    serialized_data : list[dict]

    Returns
    -------
    list
    """

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
    csv_allowed_na : list[str]
        Default ["<NA>"]. WE write all our files, so we can be more restrictive
        to reduce window for ambiguity when reading a file. In particualr, it's
        a bad idea to confuse NaN with a null value with a missing value with an
        empty value -- these are NOT the same, despite what "data science"
        conventions might suggest. If you must be awkward, try ["-NaN", "-nan",
        "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"] noting
        that "" is not in that list (that does cause problems).

    sep : str
        Default ",". Explictly define field separator

    na_rep: str
        Default "<NA>". This must be in the csv_allowed_na list. What's used as
        the default na.

    keep_default_na: bool
        Default False.

    doublequote: bool
        Default True. How we're escaping quotes in a str.

    quoting: int
        Default csv.QUOTE_NONNUMERIC. i.e. we only quote non-numeric values.

    float_precision: Literal["high", "legacy", "round_trip"]
        Default "round_trip". Weirdly, Pandas's other options, including the
        default, don't actually return what was written with floats.

    """

    csv_allowed_na: list[str] = ["<NA>"]
    sep: str = ","
    na_rep: str = "<NA>"
    keep_default_na: bool = False
    doublequote: bool = True
    quoting: int = csv.QUOTE_NONNUMERIC
    float_precision: Literal["high", "legacy", "round_trip"] = "round_trip"

    @model_validator(mode="after")
    def check_logic(self):
        if self.na_rep != "":
            if self.na_rep not in self.csv_allowed_na:
                error_msg = (
                    f"na_rep must be in csv_allowed_na. na_rep={safe_str_output(self.na_rep)};"
                    f" csv_allowed_na={safe_str_output(self.csv_allowed_na)}"
                )
                logger.error(error_msg)
                raise LookupError(error_msg)

        return self


class FIEncodingParquet(BaseModel):
    """The parameters we used for writing parquet files

    Again, there's really no need to change these.

    Attributes
    ----------
    engine : str
        Default "pyarrow". Engine to use. Has to be consistent and was tested
        with pyarrow

    index : str | None
        Default None. See
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html#pandas.DataFrame.to_parquet
    """

    engine: str = "pyarrow"
    index: str | None = None


class FIEncoding(BaseModel):
    """General encoding options, includes CSV and Parquet encoding

    Attributes
    ----------
    csv : FIEncodingCSV
        Default FIEncodingCSV(). Extra options that depend on format

    parq : FIEncodingParquet
        Default FIEncodingParquet(). Extra options that depend on format

    auto_convert_int_to_intna : bool
        Default True. Whether to automatically convert standard int dtypes to
        Pandas's Int64Dtype (which can also encode NA values), if there are one
        or more NAs or None(s) in the column

    """

    csv: FIEncodingCSV = FIEncodingCSV()
    parq: FIEncodingParquet = FIEncodingParquet()
    auto_convert_int_to_intna: bool = True


class FIBaseIndex(BaseModel):
    """Base class for our custom classes to be able to serialize/deserialize/instantiate Pandas indexes

    This is derived from Pydantic `BaseModel`, so we can (and do) use those
    facilities.
    """

    # TODO factory code to instantiate itself? (if possible from Pydantic model)

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        """Get the str name for the index (one of the FIIndex enum entires)"""

        return FIIndexType.base.name

    def get_fi_index_type(self) -> FIIndexType:
        """Get the index type (one of the FIIndex enum entires)"""

        return FIIndexType.base

    def get_as_index(self, **kwargs) -> pd.Index:
        """Creates corresponding Pandas index

        Params
        ------
        **kwargs : dict
            Not used at current time.

        Returns
        -------
        pd.Index
            The Pandas index created corresponding to our FIIndex type and data.
        """

        return pd.Index()


class FIIndex(FIBaseIndex):
    """Corresonds to pd.Index

    See https://pandas.pydata.org/docs/reference/api/pandas.Index.html

    Attributes
    ----------

    data : ArrayLike | AnyArrayLike | list | tuple
        The enumerated elements in the index.

    name : str | None = None
        Optional name.

    dtype : Dtype | DtypeObj | pd.api.extensions.ExtensionDtype | None
        Dtype of the elemenets.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: ArrayLike | AnyArrayLike | list | tuple
    name: str | None = None
    dtype: Dtype | DtypeObj | pd.api.extensions.ExtensionDtype | None

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        """Get the str name for the index (one of the FIIndex enum entires)"""

        return FIIndexType.idx.name

    def get_fi_index_type(self) -> str:
        """Get the index type (one of the FIIndex enum entires)"""

        return FIIndexType.idx

    def get_as_index(self, **kwargs) -> pd.Index:
        """Creates corresponding Pandas index

        Returns
        -------
        pd.Index
            The Pandas index created corresponding to our FIIndex type and data.
        """
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
            if (
                "data" in data.keys()
                and isinstance(data["data"], dict)
                and "el" in data["data"].keys()
                and "eltype" in data["data"].keys()
            ):
                data["data"] = _deserialize_element(data["data"])

        return data


class FIRangeIndex(FIBaseIndex):
    """Corresonds to pd.RangeIndex

    See https://pandas.pydata.org/docs/reference/api/pandas.RangeIndex.html

    Attributes
    ----------

    start : int
        Where index starts counting from.

    stop : int
        Where index stops counting.

    step : int
        Step that index counts in.

    name : str | None
        Optional name. Default None.

    dtype : DtypeObj | pd.api.extensions.ExtensionDtype | str | None
        Dtype of the index.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: int
    stop: int
    step: int
    name: str | None = None
    dtype: DtypeObj | pd.api.extensions.ExtensionDtype | str | None

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        """Get the str name for the index (one of the FIIndex enum entires)"""

        return FIIndexType.range.name

    def get_fi_index_type(self) -> str:
        """Get the index type (one of the FIIndex enum entires)"""

        return FIIndexType.range

    def get_as_index(self, **kwargs) -> pd.RangeIndex:
        """Creates corresponding Pandas index

        Returns
        -------
        pd.RangeIndex
            The Pandas index created corresponding to our FIIndex type and data.
        """

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
    """Corresonds to pd.CategoricalIndex

    See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.CategoricalIndex.html

    Attributes
    ----------

    data : ArrayLike | AnyArrayLike | list | tuple
        Elements in index.

    categories : ArrayLike | AnyArrayLike | list | tuple
        List from which elements in data must belong.

    ordered : bool
        Whether data should be ordered?

    name : str | None
        Optional name. Default None.

    dtype : DtypeObj | pd.api.extensions.ExtensionDtype | pd.CategoricalDtype | str | None
        Dtype of elements.

    """

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
        """Get the str name for the index (one of the FIIndex enum entires)"""

        return FIIndexType.categorical.name

    def get_fi_index_type(self) -> str:
        """Get the index type (one of the FIIndex enum entires)"""

        return FIIndexType.categorical

    def get_as_index(self, **kwargs) -> pd.CategoricalIndex:
        """Creates corresponding Pandas index

        Returns
        -------
        pd.CategoricalIndex
            The Pandas index created corresponding to our FIIndex type and data.
        """

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
            if (
                "data" in data.keys()
                and isinstance(data["data"], dict)
                and "el" in data["data"].keys()
                and "eltype" in data["data"].keys()
            ):
                data["data"] = _deserialize_element(data["data"])

            if (
                "categories" in data.keys()
                and isinstance(data["categories"], dict)
                and "el" in data["categories"].keys()
                and "eltype" in data["categories"].keys()
            ):
                data["categories"] = _deserialize_element(data["categories"])

        return data


class FIMultiIndex(FIBaseIndex):
    """Corresponds to pd.MultiIndex

    See https://pandas.pydata.org/docs/reference/api/pandas.MultiIndex.html and
    https://pandas.pydata.org/docs/user_guide/advanced.html

    Attributes
    ----------

    levels : list
        The number of levels in the multiindex.

    codes : list
        The list of lists (I think), of the elements in the index.

    sortorder : int | None
        Default None.

    names : list
        List of names for the levels.

    dtypes : pd.Series | list
        Dtype specifications.

    """

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
        """Get the str name for the index (one of the FIIndex enum entires)"""

        return FIIndexType.multi.name

    def get_fi_index_type(self) -> str:
        """Get the index type (one of the FIIndex enum entires)"""

        return FIIndexType.multi

    def get_as_index(self, **kwargs) -> pd.MultiIndex:
        """Creates corresponding Pandas index

        Returns
        -------
        pd.MultiIndex
            The Pandas index created corresponding to our FIIndex type and data.
        """

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
    """Corresponds to pd.IntervalIndex

    See https://pandas.pydata.org/docs/reference/api/pandas.IntervalIndex.html

    Attributes
    ----------

    data : pd.arrays.IntervalArray | np.ndarray
        The data array (of intervals!).

    closed : IntervalClosedType
        How each interval is closed or not: "left", "right", "closed", "neither".

    name : str or None
        Optional name. Default None.

    dtype : pd.IntervalDtype | str | None

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.arrays.IntervalArray | np.ndarray
    closed: IntervalClosedType
    name: str | None = None
    dtype: pd.IntervalDtype | str | None

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        """Get the str name for the index (one of the FIIndex enum entires)"""

        return FIIndexType.interval.name

    def get_fi_index_type(self) -> str:
        """Get the index type (one of the FIIndex enum entires)"""

        return FIIndexType.interval

    def get_as_index(self, **kwargs) -> pd.IntervalIndex:
        """Creates corresponding Pandas index

        Returns
        -------
        pd.IntervalIndex
            The Pandas index created corresponding to our FIIndex type and data.
        """

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
    """Corresponds to pd.DatetimeIndex

    See https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html

    Attributes
    ----------

    data: ArrayLike | AnyArrayLike | list | tuple
        Array of datetimes.

    freq: _Frequency | None = None
        Optional frequency. See Pandas docs for what this means.

    tz: tzinfo | str | None
        Optional tz.

    name: str | None = None
        Optional name.

    dtype: Dtype | str | None

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: ArrayLike | AnyArrayLike | list | tuple
    freq: _Frequency | None = None
    tz: tzinfo | str | None  # most what it should be from pandas src
    name: str | None = None
    dtype: Dtype | str | None  # Hmmm.

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        """Get the str name for the index (one of the FIIndex enum entires)"""

        return FIIndexType.datetime.name

    def get_fi_index_type(self) -> str:
        """Get the index type (one of the FIIndex enum entires)"""

        return FIIndexType.datetime

    def get_as_index(self, **kwargs) -> pd.DatetimeIndex:
        """Creates corresponding Pandas index

        Returns
        -------
        pd.DatetimeIndex
            The Pandas index created corresponding to our FIIndex type and data.
        """

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
    """Corresponds to pd.TimedeltaIndex

    See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.TimedeltaIndex.html

    Attributes
    ----------

    data : ArrayLike | AnyArrayLike | list | tuple
        Array of timedeltas.

    freq : str | BaseOffset | None = None
        Optional frequency. See Pandas docs for details.

    name : str | None = None
        Optional name.

    dtype : DtypeObj | np.dtypes.TimeDelta64DType | Literal["<m8[ns]"] | str | None

    """

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
        """Get the str name for the index (one of the FIIndex enum entires)"""

        return FIIndexType.timedelta.name

    def get_fi_index_type(self) -> str:
        """Get the index type (one of the FIIndex enum entires)"""

        return FIIndexType.timedelta

    def get_as_index(self, **kwargs) -> pd.TimedeltaIndex:
        """Creates corresponding Pandas index

        Returns
        -------
        pd.TimedeltaIndex
            The Pandas index created corresponding to our FIIndex type and data.
        """

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
            if (
                "data" in data.keys()
                and isinstance(data["data"], dict)
                and "el" in data["data"].keys()
                and "eltype" in data["data"].keys()
            ):
                data["data"] = _deserialize_element(data["data"])

        return data


class FIPeriodIndex(FIBaseIndex):
    """Corresponds to pd.PeriodIndex

    See https://pandas.pydata.org/docs/reference/api/pandas.PeriodIndex.html

    data: ArrayLike | AnyArrayLike | list | tuple
        Array of periods.

    freq: _Frequency | None = None
        Optional frequency. See Pandas docs.

    name: str | None = None
        Optional name

    dtype: DtypeObj | pd.PeriodDtype | str | None  # Hmmm.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: ArrayLike | AnyArrayLike | list | tuple
    freq: _Frequency | None = None
    name: str | None = None
    dtype: DtypeObj | pd.PeriodDtype | str | None  # Hmmm.

    @computed_field(title="index_type")
    @property
    def index_type(self) -> str:
        """Get the str name for the index (one of the FIIndex enum entires)"""

        return FIIndexType.period.name

    def get_fi_index_type(self) -> str:
        """Get the index type (one of the FIIndex enum entires)"""

        return FIIndexType.period

    def get_as_index(self, **kwargs) -> pd.PeriodIndex:
        """Creates corresponding Pandas index

        Returns
        -------
        pd.PeriodIndex
            The Pandas index created corresponding to our FIIndex type and data.
        """

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
            if (
                "data" in data.keys()
                and isinstance(data["data"], dict)
                and "el" in data["data"].keys()
                and "eltype" in data["data"].keys()
            ):
                data["data"] = _deserialize_element(data["data"])

        return data


class FIMetainfo(BaseModel):
    """All the collected metadata we use when saving or loading

    N.B. The _order_ of the attributes is important in the sense that the
    serialization automatically preserves the order, and then `yaml.dump()` does
    too. This means we can make the YAML file a little easier to read/parse by a
    human.

    Attributes
    ----------

    datafile : Path
        Ironically, this should always just be the filename with no paths

    file_format : FIFileFormatEnum
        The file format of datafile.

    format_version: int
        Default 1. Not really used yet but we might need to version the YAML file.

    hash: str | None
        SHA256 hash of the datafile.

    encoding: FIEncoding
        How the datafile was or is to be encoded.

    custom_info: SerializeAsAny[FIBaseCustomInfo]
        Structured custom info. Can just be an empty FIBaseCustomInfo object.

    serialized_dtypes: dict
        Dtypes of the dataframe.

    index: FIBaseIndex
        Index information encoded as a FIBaseIndex object (descendent thereof).

    columns: FIBaseIndex
        Columns, again, specified as an FIIndex object

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Ironically, this should always just be the filename with no paths
    datafile: Path

    # File format
    file_format: FIFileFormatEnum

    # Format version
    format_version: int = 1

    # SHA256 hash
    hash: str | None = None

    # Encoding
    encoding: FIEncoding

    # Custom info (user defined metainfo)
    custom_info: SerializeAsAny[FIBaseCustomInfo]

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

    @field_validator("custom_info", mode="before")
    @classmethod
    def validator_custom_info(
        cls, value: dict | FIBaseCustomInfo, info: ValidationInfo
    ) -> FIBaseCustomInfo:
        # Shortcut exit, if we've been passed something with extra_info already
        # instantiated. We only deal with dicts here.
        if not isinstance(value, dict):
            return value

        # By default we don't use a context
        clss_custom_info = None

        # If we don't have context, just use the base class or return as-is
        if info.context and isinstance(info.context, dict):
            # Get the available classes for extra_info (this should also be a
            # dictionary)
            clss_custom_info = info.context.get(
                "clss_custom_info", {"FIBaseCustomInfo": FIBaseCustomInfo}
            )
            assert isinstance(clss_custom_info, dict)

        # Now process
        value_classname = value.get("classname", None)
        if (
            value_classname
            and clss_custom_info is not None
            and value_classname in clss_custom_info.keys()
        ):
            # Now instantiate the model
            custom_info_class = clss_custom_info[value_classname]
        elif value_classname in globals().keys() and issubclass(
            globals()[value_classname], FIBaseCustomInfo
        ):
            custom_info_class = globals()[value_classname]
        else:
            error_msg = f"Neither context for supplied classname nor is it a subclass of FIBaseCustomInfo. classname={safe_str_output(value_classname)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        assert issubclass(custom_info_class, FIBaseCustomInfo)
        return custom_info_class.model_validate(value, context=info.context)

    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data: Any) -> Any:
        # TODO perhaps move index and columns into separate field validators.

        if isinstance(data, dict):
            # Need to ensure the index and columns and custominfo are created as
            # the correct object type, not just instantiating the base class.
            if "index" in data.keys() and isinstance(data["index"], dict):
                data["index"] = _deserialize_index_dict_to_fi_index(data["index"])

            if "columns" in data.keys() and isinstance(data["columns"], dict):
                data["columns"] = _deserialize_index_dict_to_fi_index(data["columns"])

        return data


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

    if extension == ".csv":
        return FIFileFormatEnum.csv
    elif extension == ".parq" or extension == ".parquet":
        return FIFileFormatEnum.parquet
    else:
        error_msg = (
            f"Couldn't find FIFileFormatEnum for {safe_str_output(extension)}."
            f" Filename={safe_str_output(datafile.name, 1024)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _check_metafile_name(
    datafile: Path,
    metafile: Path | None = None,
) -> Path:
    """Checks or generates the absolute path to the metafile

    Parameters
    ----------
    datafile : Path
        Absolute path to the datafile.
    metafile : Path | None, optional
        Optional metafile name (will check correctness). If None, will generate and return.

    Returns
    -------
    Path
        Absolute path to metafile.
    """

    # Determine output metafile name and ensure extension is correct
    if metafile is None:
        loc_metafile = datafile.with_suffix(".yaml")
    else:
        loc_metafile = metafile
        if loc_metafile.suffix not in [".yaml", ".yml"]:
            error_msg = f"File extension for metadata file must be .yaml. Got {safe_str_output(loc_metafile.suffix)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Check path for datafile and metafile have the same parent directory
    if len(loc_metafile.parents) <= 1:
        # In this instance, loc_metafile is just a filename so we can prefix the dir from datafile
        loc_metafile = datafile.parent / loc_metafile
    elif datafile.parent != loc_metafile.parent:
        error_msg = f"Path for datafile and metafile must be the same. datafile={safe_str_output(datafile)}, loc_metafile={safe_str_output(metafile)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return loc_metafile


def _preprocess_common(df: pd.DataFrame, encoding: FIEncoding):
    """Preprocess checks that do not alter the dataframe

    Preprocess alterations have to happen in either `_preprocess_inplace()` or
    `_preprocess_safe()`.
    """

    # Check column index has unique elements
    if not df.columns.is_unique:
        error_msg = "Column names must be unique! (This is a tighter restriction than Pandas but avoids a lot of stupid mistakes, so we enforce it)"
        logger.error(error_msg)
        raise ValueError(error_msg)


def _preprocess_inplace(df: pd.DataFrame, encoding: FIEncoding):
    """Pre-process dataframe inplace, i.e. notionally 'unsafe'

    In practice though, we put all the alterations in here and if running in
    safe mode we'd be fed a copy.

    Parameters
    ----------
    df : pd.DataFrame
    encoding : FIEncoding
    """

    _preprocess_common(df, encoding)


def _preprocess_safe(df: pd.DataFrame, encoding: FIEncoding) -> pd.DataFrame:
    """Pre-process dataframe safely, i.e. make a copy first

    This actually just calls `_preprocess_inplace()` to run on a copy, so put
    any actual logic in there. This is just a copy operation.

    Parameters
    ----------
    df : pd.DataFrame
    encoding : FIEncoding

    Returns
    -------
    pd.DataFrame
    """

    _preprocess_common(df, encoding)

    # Make a copy of the dataframe
    loc_df = copy.deepcopy(df)

    # Modify inplace the copied dataframe
    _preprocess_inplace(loc_df, encoding)

    return loc_df


def _serialize_df_dtypes_to_dict(df: pd.DataFrame) -> dict:
    """Serializes the dtypes from a dataframe into a dictionary

    This isn't quite as obvious as it seems because the column label isn't
    necessarily a string. Indeed, it can be a tuple (if a MultiIndex is used for
    columns).

    Params
    ------
    df : pd.DataFrame

    Returns
    -------
    dict

    """

    serialized_dtypes = {}

    # Loop through and just append to serialized_dtypes unless we get a special
    # type we must manually serialize, e.g. category.
    for col_name, dtype in df.dtypes.to_dict().items():
        # We do have to serialise the key (column name) somehow. Otherwise
        # things like tuples end up being encoded in the YAML file in an
        # undesirable way. BUT don't want to use usual serialization for the
        # keys as it will become less readable+difficult. So lets try simple
        # string representation for now and store properly serialized version as
        # a value.
        loc_col_name = str(col_name)

        if dtype == "category":
            dtype_full = df.dtypes[col_name]
            assert isinstance(dtype_full, pd.CategoricalDtype)
            serialized_dtypes[loc_col_name] = {"dtype_str": str(dtype)}
            serialized_dtypes[loc_col_name]["categories"] = (
                dtype_full.categories.to_list()
            )
            serialized_dtypes[loc_col_name]["ordered"] = str(dtype_full.ordered)
        else:
            serialized_dtypes[loc_col_name] = {"dtype_str": str(dtype)}

        serialized_dtypes[loc_col_name]["serialized_col_name"] = _serialize_element(
            col_name
        )

    return serialized_dtypes


def _deserialize_df_types(serialized_dtypes: dict) -> dict:
    """Deserializes output from `_serialize_df_dtypes_to_dict()`

    Parameters
    ----------
    serialized_dtypes : dict

    Returns
    -------
    dict
    """

    deserialized_dtypes = {}
    for col in serialized_dtypes:
        # Get deserialized column name
        assert "serialized_col_name" in serialized_dtypes[col]
        loc_col_name = _deserialize_element(
            serialized_dtypes[col]["serialized_col_name"]
        )

        # Get string representation of dtype
        dtype_str = serialized_dtypes[col].get("dtype_str", None)
        if dtype_str is None:
            error_msg = f"Got column in serialized dtypes without a dtype_str field. col={safe_str_output(col)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

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


def _deserialize_dtypes_for_read_csv(serialized_dtypes: dict) -> dict:
    """Deserialize dtypes from what's stored in FIMetaInfo for use with Pandas `read_csv()`

    Parameters
    ----------
    serialized_dtypes : dict

    Returns
    -------
    dict
        Deserialized dict appropriate for Pandas's consumption
    """

    deserialized_dtypes = {}
    for col in serialized_dtypes:
        # There appears to be serious difficulties when referring to columns. We
        # can try to deal with this here. N.B. This seems necessary to make it
        # work! i.e. forcing reference to columns to be a string.
        if False:
            pass
        else:
            # By default make it a string
            loc_col = str(col)

        # Get string representation of dtype
        dtype_str = serialized_dtypes[col].get("dtype_str", None)
        if dtype_str is None:
            error_msg = f"Got column in serialized dtypes without a dtype_str field. col={safe_str_output(col)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Try to apply dtypes. Usually we just "pass through" but there are some
        # that need a more permissive default (str or object) so that the column
        # can then be manually converted later.
        if dtype_str == "category":
            deserialized_dtypes[loc_col] = "object"
        elif dtype_str[0:10] == "datetime64":
            # Catching all datetime64...
            deserialized_dtypes[loc_col] = "object"
        elif dtype_str[0:11] == "timedelta64":
            # Catching all timedelta64
            deserialized_dtypes[loc_col] = "object"
        elif dtype_str[0:8].lower() == "interval":
            # Catching all interval variations: https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
            # deserialized_dtypes[col] = "Interval"
            pass
        else:
            deserialized_dtypes[loc_col] = dtype_str

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
            if dtype_info["dtype_str"] == "timedelta64[ns]":
                df[col] = df[col].astype(dtype_info["dtype_str"])
            else:
                df[col] = df[col].astype(dtype_info["dtype_str"])

    return None


def _serialize_index_to_metainfo_index(idx: pd.Index) -> FIBaseIndex:
    """Serializes a Pandas index into one of our FI*Index classes

    Parameters
    ----------
    idx : pd.Index
        The index, e.g. a pd.MultiIndex, to serialize to one of our FI*Index
        objects.

    Returns
    -------
    FIBaseIndex
        The instantiated object, which will be a descendent of FIBaseIndex.
    """

    if not isinstance(idx, pd.Index):
        error_msg = f"Must pass a Pandas Index. Got a {type(idx)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

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
        raise TypeError(error_msg)


def _deserialize_index_dict_to_fi_index(index: dict) -> FIBaseIndex:
    """Deserializes an index stored as a dictionary (from YAML file) into on of our FI*Indexes

    Parameters
    ----------
    index : dict
        The dictionary that is storing the index.

    Returns
    -------
    FIBaseIndex
        The instantiated object, which will be a descendent of the FIBaseIndex class.
    """

    index_type = index.get("index_type", None)
    if index_type is None or index_type == "":
        error_msg = "index_type cannot be empty or None when deserializing."
        logger.error(error_msg)
        raise ValueError(error_msg)

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
        error_msg = (
            f"index_type not recognised. index_type={safe_str_output(index_type)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _compile_metainfo(
    datafile: Path,
    file_format: FIFileFormatEnum,
    hash: str,
    encoding: FIEncoding,
    custom_info: FIBaseCustomInfo,
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
    custom_info : FIBaseCustomInfo
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
        custom_info=custom_info,
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
    dtypes: dict,
    num_index_cols: int = 1,
    num_index_rows: int = 1,
) -> pd.DataFrame:
    """Read from CSV file using supplied options

    Parameters
    ----------
    input_datafile : Path
        What CSV file to read from.
    encoding : FIEncoding
        The encoding options to use (like sep, etc).
    dtypes : dict
        Dictionary of dtype specifications.
    num_index_cols : int, optional
        How many columns as used for the row index. Always the first n columns.
    num_index_rows : int, optional
        How many rows are used to specify the columns. Always the first n rows.

    Returns
    -------
    pd.DataFrame
    """

    # We always put our index column(s) first (row labels)
    index_col = list(range(0, num_index_cols))

    # And we always put our index rows first (column labels)
    index_row = list(range(0, num_index_rows))

    # Get dtypes so that we can specify for `read_csv()`
    deserialized_dtypes = _deserialize_dtypes_for_read_csv(dtypes)

    df = pd.read_csv(
        input_datafile,
        header=index_row,
        index_col=index_col,
        float_precision=encoding.csv.float_precision,
        doublequote=encoding.csv.doublequote,
        sep=encoding.csv.sep,
        dtype=deserialized_dtypes,
        keep_default_na=encoding.csv.keep_default_na,
        na_values=encoding.csv.csv_allowed_na,
    )

    return df


def _write_to_parquet(df: pd.DataFrame, datafile: Path, encoding: FIEncoding):
    """Write to Parquet datafile using supplied options

    Parquet does not permit columns with different dtypes, i.e. it'll bork on
    the row index if it's not the same dtype in each entry. It complains if one
    does similar with the column index. So we don't write these, i.e. set
    `index=False` option.

    Parameters
    ----------
    df : pd.DataFrame
    datafile : Path
    encoding : FIEncoding
    """
    df.to_parquet(datafile, engine="pyarrow", index=False)


def _read_from_parquet(input_datafile: Path, encoding: FIEncoding) -> pd.DataFrame:
    """Read from Parquet file

    This shouldn't have column index or row index, as we don't save them (see
    above).

    Parameters
    ----------
    input_datafile : Path
    encoding : FIEncoding

    Returns
    -------
    pd.DataFrame
    """

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
        h_targetfile.write(f"# Metadata for {safe_str_output(datafile, 1024)}\n")
        h_targetfile.write("---\n\n")

        # Write out the rest of the file now
        h_targetfile.write(yaml_output)


def _read_metafile(metafile: Path, context: dict | None = None) -> FIMetainfo:
    """Reads in metainfo from file

    Parameters
    ----------
    metafile : Path
        The location of the metafile.
    context : dict | None, optional
        If manually supplying a context to decode the structured custom info, by
        default None (in which was subclass type checks are used).

    Returns
    -------
    FIMetainfo
    """

    # Read metainfo from file
    with open(metafile, "r") as h_metafile:
        metainfo_dict = yaml.load(h_metafile, Loader=Loader)

    # Convert into an FIMetainfo model
    metainfo = FIMetainfo.model_validate(metainfo_dict, context=context)

    return metainfo


def write_df_to_file(
    df: pd.DataFrame,
    datafile: Path | str,
    metafile: Path | str | None = None,
    file_format: FIFileFormatEnum | Literal["csv", "parquet"] | None = None,
    encoding: FIEncoding | None = None,
    custom_info: FIBaseCustomInfo | dict = {},
    preprocess_inplace=True,
) -> Path:
    """Writes a dataframe to file

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to save.
    datafile : Path or str
        The datafile to save the dataframe to.
    metafile : Path or str or None (optional)
        Metafile name, can be only the filename or with a path (which must be
        the same as for datafile). If not supplied or None, will be determined
        automatically.
    file_format : FIFileFormatEnum | Literal['csv', 'parquet'] | None
        The file format. If not supplied will be determined automatically.
    encoding : FIEncoding | None, optional
        Datafile encoding options.
    custom_info : FIBaseCustomInfo or dict
        Custom user metadata to be stored. IF supplied as a FIBaseCustomInfo (or
        descendent) then it stores things properly. If supplied as a dict, then
        will create a FIBaseCustomInfo class and store the dictionary in the
        `unstructured_data` field.
    preprocess_inplace : bool, optional

    Returns
    -------
    Path
        A Path object with the metainfo filename in it.

    """

    # Check types and existence correct for datafile and metafile
    if not isinstance(datafile, (Path, str)):
        error_msg = f"datafile must be a Path or str. Got type={type(datafile)}, value={safe_str_output(datafile)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if metafile is not None and not isinstance(metafile, (Path, str)):
        error_msg = "When metafile given (not None), it must be a Path or str."
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Cast datafile and metafile to str
    if isinstance(datafile, str):
        datafile = Path(datafile)

    if isinstance(metafile, str):
        metafile = Path(metafile)

    # Determine output format
    if file_format is None:
        loc_file_format = _detect_file_format_from_filename(datafile)
    else:
        loc_file_format = FIFileFormatEnum(file_format)

    # Determine metafile name
    loc_metafile = _check_metafile_name(datafile, metafile)

    # If we've got encoding parameters, use them; otherwise use defaults
    if encoding is None:
        encoding = FIEncoding()

    # Preprocess
    if preprocess_inplace:
        _preprocess_inplace(df, encoding)
        loc_df = df
    else:
        loc_df = _preprocess_safe(df, encoding)

    # Deal with custom info situation
    if isinstance(custom_info, dict):
        # We create FIBaseCustomInfo ourselves, and assign the dictionary into
        # unstructured_data
        loc_custom_info = FIBaseCustomInfo.model_validate(
            {"unstructured_data": custom_info}
        )
    elif isinstance(custom_info, FIBaseCustomInfo):
        loc_custom_info = custom_info
    else:
        raise TypeError("custom_info must be a dict or descendent of FIBaseCustomInfo")

    # Write to the data file
    if loc_file_format == FIFileFormatEnum.csv:
        _write_to_csv(loc_df, datafile, encoding)
    elif loc_file_format == FIFileFormatEnum.parquet:
        _write_to_parquet(loc_df, datafile, encoding)
    else:
        error_msg = "Output format not supported. This shouldn't happen."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Commented out because want to support Python 3.10 for a while, and
    # hashlib.file_digest() didn't appear until 3.11.
    # with open(datafile, "rb") as h_datafile:
    #     digest = hashlib.file_digest(h_datafile, "sha256")
    # hash = digest.hexdigest()

    # Version safe version of file hashing (SHA256)
    hash = hash_file(datafile)

    # Compile all the metainfo into a dictionary
    metainfo = _compile_metainfo(
        datafile=datafile,
        file_format=loc_file_format,
        hash=hash,
        encoding=encoding,
        custom_info=loc_custom_info,
        df=loc_df,
    )

    # Write metafile
    _write_metafile(datafile, loc_metafile, metainfo)

    return loc_metafile


def write_df_to_csv(
    df: pd.DataFrame,
    datafile: Path | str,
    encoding: FIEncoding | None = None,
    custom_info: FIBaseCustomInfo | dict = {},
    preprocess_inplace=True,
) -> Path:
    """Simplified wrapper around `write_df_to_file()` to write dataframe to CSV

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe.
    datafile : Path or str
        Target datafile.
    encoding : FIEncoding | None, optional
        Encoding specs, can be left None for defaults.
    custom_info : dict, optional
        Any custom meta data.
    preprocess_inplace : bool, optional
        Whether to do preprocessing inplace (might modify original), by default True

    Returns
    -------
    Path
        A Path object with the metainfo filename in it.
    """

    return write_df_to_file(
        df=df,
        datafile=datafile,
        metafile=None,
        file_format=FIFileFormatEnum.csv,
        encoding=encoding,
        custom_info=custom_info,
        preprocess_inplace=preprocess_inplace,
    )


def write_df_to_parquet(
    df: pd.DataFrame,
    datafile: Path | str,
    encoding: FIEncoding | None = None,
    custom_info: FIBaseCustomInfo | dict = {},
    preprocess_inplace=True,
) -> Path:
    """Simplified wrapper around `write_df_to_file()` to write dataframe to CSV

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe.
    datafile : Path or str
        Target datafile.
    encoding : FIEncoding | None, optional
        Encoding specs, can be left None for defaults.
    custom_info : dict, optional
        Any custom meta data.
    preprocess_inplace : bool, optional
        Whether to do preprocessing inplace (might modify original), by default True

    Returns
    -------
    Path
        A Path object with the metainfo filename in it.
    """

    return write_df_to_file(
        df=df,
        datafile=datafile,
        metafile=None,
        file_format=FIFileFormatEnum.parquet,
        encoding=encoding,
        custom_info=custom_info,
        preprocess_inplace=preprocess_inplace,
    )


def read_df(
    metafile: Path | str,
    strict_hash_check: bool = True,
    context_metainfo: dict | None = None,
) -> tuple[pd.DataFrame, FIMetainfo]:
    """Load a dataframe from file

    Supply the metainfo filename, not the datafilename.

    Parameters
    ----------
    metafile : Path
        The YAML file that is associated with the datafile.
    strict_hash_check : bool, optional
        Whether we raise an exception if the hash is wrong.
    context_metainfo : dict | None, optional
        If manually supplying a context to decode the structured custom info, by
        default None (in which was subclass type checks are used).

    Returns
    -------
    tuple[pd.DataFrame, FIMetainfo]:
        A tuple with the dataframe and the metainfo object.
    """

    # Check metafile not empty and correct types
    if not isinstance(metafile, (Path, str)):
        error_msg = f"metafile must be a Path or str. Got type={type(metafile)}, value={safe_str_output(metafile)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if isinstance(metafile, str):
        metafile = Path(metafile)

    # Load metainfo
    metainfo = _read_metafile(metafile, context=context_metainfo)

    # Check datafile's hash
    datafile_abs = Path(metafile.parent / metainfo.datafile).resolve()
    # Commented out to support Python 3.10 for a while (hashlib.file_digest() didn't appear until 3.11)
    # with open(datafile_abs, "rb") as h_datafile:
    #     digest = hashlib.file_digest(h_datafile, "sha256")
    # hash = digest.hexdigest()
    hash = hash_file(datafile_abs)

    if hash != metainfo.hash:
        error_msg = f"Hash comparison failed. metainfo.hash={safe_str_output(metainfo.hash)}, calcualted hash={safe_str_output(hash)}."
        if strict_hash_check:
            logger.error(error_msg)
            raise ValueError(error_msg)
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

    # Load the data
    if metainfo.file_format == FIFileFormatEnum.csv:
        df = _read_from_csv(
            datafile_abs,
            metainfo.encoding,
            dtypes=metainfo.serialized_dtypes,
            num_index_cols=num_index_cols,
            num_index_rows=num_index_rows,
        )
    elif metainfo.file_format == FIFileFormatEnum.parquet:
        df = _read_from_parquet(datafile_abs, metainfo.encoding)
    else:
        error_msg = f"Input format ({safe_str_output(metainfo.file_format)}) not supported. We only support CSV and Parquet."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Apply index and columns
    df.index = metainfo.index.get_as_index()
    df.columns = metainfo.columns.get_as_index()

    # Apply dtypes
    _apply_serialized_dtypes(df, metainfo.serialized_dtypes)

    return (df, metainfo)
