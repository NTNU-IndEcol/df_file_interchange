"""Examples, mainly used for tests"""

import re

import numpy as np
import pandas as pd


def generate_example_indices():
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

    fi_rangeindex_1 = pd.RangeIndex(start=-10, stop=-5, step=1)
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
    fi_intervalindex_4 = pd.interval_range(
        start=-3.0,
        end=11.0,
        freq=0.5,   # type: ignore
        closed="neither",
        name="intervalindex4",
    )

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

    fi_periodindex_1 = pd.PeriodIndex.from_fields(  # type: ignore   (for some reason, pylance can't see .from_fields())
        year=[2000, 2002, 2004], quarter=[1, 3, 2]
    )
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


def generate_dfs_from_indices(test_indices):
    dfs = {}
    for k, v in test_indices.items():
        idx_len = len(v)

        # Create a dataframe with index used both as row index and column index.
        # Except for categorical, which would not be unique.
        if re.match(".*categorical.*", k):
            df = pd.DataFrame(
                np.random.randn(idx_len, idx_len),
                index=v,
                columns=pd.RangeIndex(0, idx_len),
            )
        else:
            df = pd.DataFrame(np.random.randn(idx_len, idx_len), index=v, columns=v)

        dfs[k] = df

    return dfs


def generate_example_1(b_include_complex: bool = False):
    """Generate an example dataframe for test purposes

    Parameters
    ----------
    b_include_complex : bool, optional
        Whether to include complex number examples, which fail when writing to
        Parquet using the Arrow engine. It's fine for CSV though.

    Returns
    -------
    pd.DataFrame
    """

    # See
    # https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
    # https://pandas.pydata.org/docs/reference/arrays.html#api-arrays-datetime

    dates = pd.date_range("1/1/2000", periods=5, name="Date")
    df = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=["A", "B", "C"])

    # NumPy data types
    # ----------------

    # Signed ints
    df["F_np_int8"] = pd.array(
        [1, 0, -50, np.iinfo(np.int8).min, np.iinfo(np.int8).max], dtype="int8"
    )
    df["F_np_int16"] = pd.array(
        [1, 0, -50, np.iinfo(np.int16).min, np.iinfo(np.int16).max], dtype="int16"
    )
    df["F_np_int32"] = pd.array(
        [1, 0, -50, np.iinfo(np.int32).min, np.iinfo(np.int32).max], dtype="int32"
    )
    df["F_np_int64"] = pd.array(
        [1, 0, -50, np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype="int64"
    )
    df["F_np_longlong"] = pd.array(
        [1, 0, -50, np.iinfo(np.longlong).min, np.iinfo(np.longlong).max],
        dtype="longlong",
    )

    # Unsigned ints
    df["F_np_uint8"] = pd.array(
        [1, 0, 50, np.iinfo(np.uint8).min, np.iinfo(np.uint8).max], dtype="uint8"
    )
    df["F_np_uint16"] = pd.array(
        [1, 0, 50, np.iinfo(np.uint16).min, np.iinfo(np.uint16).max], dtype="uint16"
    )
    df["F_np_uint32"] = pd.array(
        [1, 0, 50, np.iinfo(np.uint32).min, np.iinfo(np.uint32).max], dtype="uint32"
    )
    df["F_np_uint64"] = pd.array(
        [1, 0, 50, np.iinfo(np.uint64).min, np.iinfo(np.uint64).max], dtype="uint64"
    )
    df["F_np_ulonglong"] = pd.array(
        [1, 0, 50, np.iinfo(np.ulonglong).min, np.iinfo(np.ulonglong).max],
        dtype="ulonglong",
    )

    # Floats

    # Apparently float16 isn't supported in Pandas. See https://github.com/pandas-dev/pandas/issues/56361
    # df["F_np_float16"] = pd.array(
    #     [1.0, -np.pi, np.NaN, np.finfo(np.float16).min, np.finfo(np.float16).max],
    #     dtype="float16",
    # )
    df["F_np_float32"] = pd.array(
        [1.0, -np.pi, np.NaN, np.finfo(np.float32).min, np.finfo(np.float32).max],
        dtype="float32",
    )
    df["F_np_float64"] = pd.array(
        [1.0, -np.pi, np.NaN, np.finfo(np.float64).min, np.finfo(np.float64).max],
        dtype="float64",
    )

    # Complex (shouldn't do this if writing Parquet with Arrow engine)
    if b_include_complex:
        df["F_np_complex64"] = pd.array(
            [1.0 + 1.0j, -2.0 - 1j, np.pi * 1j, np.NaN, 5.0 - 800j], dtype="complex64"
        )
        df["F_np_complex128"] = pd.array(
            [1.0 + 1.0j, -2.0 - 1j, np.pi * 1j, np.NaN, 5.0 - 800j], dtype="complex128"
        )
        df["F_np_clongdouble"] = pd.array(
            [1.0 + 1.0j, -2.0 - 1j, np.pi * 1j, np.NaN, 5.0 - 800j], dtype="clongdouble"
        )

    # bools
    df["F_np_bool"] = pd.array([True, False, True, True, False], dtype="bool")

    # datetimes
    df["F_np_datetime64"] = pd.array(
        [
            np.datetime64("2010-01-31T10:23:01"),
            np.datetime64("nAt"),
            np.datetime64(0, "ns"),
            np.datetime64(2**63 - 1, "ns"),
            np.datetime64(-(2**63) + 1, "ns"),
        ],
        dtype="datetime64[ns]",
    )

    # timedeltas
    # df["F_np_timedelta64"] = pd.array(
    #     [
    #         np.timedelta64(1, "D"),
    #         np.timedelta64("nAt"),
    #         np.timedelta64(2**63 - 1, "ns"),
    #         np.timedelta64(-(2**63) + 1, "ns"),
    #         np.timedelta64(0, "s"),
    #     ],
    #     dtype="timedelta64[ns]",
    # )
    df["F_np_timedelta64"] = pd.array(
        [
            np.timedelta64(1, "D"),
            np.timedelta64("nAt"),
            np.timedelta64(2**63 - 1, "ns"),
            np.timedelta64(
                -(2**62), "ns"
            ),  # TODO: this should work with -(2**63)+1 but it doesn't. Don't know why.
            np.timedelta64(0, "s"),
        ],
        dtype="timedelta64[ns]",
    )

    # Pandas types / pandas extended data types
    # Note so can find later: https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
    # https://pandas.pydata.org/docs/reference/arrays.html
    # ----------

    # Careful with syntax! See, https://github.com/pandas-dev/pandas/issues/57644
    df["F_pd_DatetimeTZDtype"] = pd.array(
        [
            "2010/01/31 10:23:01",
            None,
            0,
            np.datetime64(2**63 - 1, "ns"),
            np.datetime64(-(2**62) + 1, "ns"),
        ],  # TODO again, the lower limit is using 62 instead of 63
        dtype="datetime64[ns, Europe/Paris]",
    )

    # This is probably redunant
    df["F_pd_Timestamp"] = pd.array(
        [
            pd.Timestamp("2010/01/31 10:23:01"),
            pd.Timestamp("NaT"),
            pd.Timestamp(0),
            pd.Timestamp(2**63 - 1),
            pd.Timestamp(-(2**62) + 1),
        ],
    )

    # Timedeltas (again, probably redundant)
    df["F_pd_Timedelta"] = pd.array(
        [
            np.timedelta64(1, "D"),
            np.timedelta64("nAt"),
            np.timedelta64(2**63 - 1, "ns"),
            np.timedelta64(
                -(2**62), "ns"
            ),  # TODO: this should work with -(2**63)+1 but it doesn't. Don't know why.
            np.timedelta64(0, "s"),
        ],
    )

    # Missing: PeriodDtype

    # IntervalDtype
    # df["F_pd_IntervalDtype"] = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5), pd.Interval(2, 5), pd.Interval(3, 5), pd.Interval(4, 5)])

    # Int64Dtype
    df["F_pd_Int64Dtype"] = pd.array(
        [1, None, 0, np.iinfo(np.int64).min, np.iinfo(np.int64).max],
        dtype=pd.Int64Dtype(),
    )

    # Float64Dtype
    df["F_pd_Float64Dtype"] = pd.array(
        [1.0, np.pi, None, np.finfo(np.float64).min, np.finfo(np.float64).max],
        dtype=pd.Float64Dtype(),
    )

    # CategoricalDtype
    df["F_pd_CategoricalDtype"] = pd.Categorical(
        ["a", "b", "c", "a", "a"], categories=["a", "b", "c"], ordered=True
    )

    # StringDtype
    df["F_pd_StringDtype"] = pd.array(
        ["this", "col", "is string", "based", "!"], dtype=pd.StringDtype()
    )

    # Missing: Sparse

    # BooleanDtype
    df["F_pd_BooleanDtype"] = pd.array(
        [True, False, None, True, False], dtype=pd.BooleanDtype()
    )

    # Missing: ArrowDtype

    return df
