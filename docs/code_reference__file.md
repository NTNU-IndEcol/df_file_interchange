# Code Reference

## df_file_interchange.file.rw

The classes and functions in this module do the writing and reading.

### Premable

::: df_file_interchange.file.rw.chk_strict_frames_eq_ignore_nan

::: df_file_interchange.file.rw.FIFileFormatEnum

::: df_file_interchange.file.rw.FIIndexType


### Encoding Specifications

Encoding options can be specified in a `FIEncoding` object, which in turn contains `FIEncodingCSV` and `FIEncodingParquet` as attributes (only the object corresponding to the file format applies when writing). These all construct themselves with default options, it's usually ill-advised to change these.

::: df_file_interchange.file.rw.FIEncodingCSV

::: df_file_interchange.file.rw.FIEncodingParquet

::: df_file_interchange.file.rw.FIEncoding

### Our Index Representation(s)

We have our own classes to represent Pandas indexes, which can perform operations such as serialization and instantiation (of the Pandas index). Everything here should derive from the `FIBaseIndex` base class.

::: df_file_interchange.file.rw.FIBaseIndex

::: df_file_interchange.file.rw.FIIndex

::: df_file_interchange.file.rw.FIRangeIndex

::: df_file_interchange.file.rw.FICategoricalIndex

::: df_file_interchange.file.rw.FIMultiIndex

::: df_file_interchange.file.rw.FIIntervalIndex

::: df_file_interchange.file.rw.FIDatetimeIndex

::: df_file_interchange.file.rw.FITimedeltaIndex

::: df_file_interchange.file.rw.FIPeriodIndex

::: df_file_interchange.file.rw.FIMetainfo


### The Write and Read Functions

These are what are exposed to the user, to roundtrip write and read dataframes.

::: df_file_interchange.file.rw.write_df_to_file

::: df_file_interchange.file.rw.write_df_to_csv

::: df_file_interchange.file.rw.read_df