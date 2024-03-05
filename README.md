# df_file_interchange

This package is designed to store complete specification DataFrames (including indexes) along with custom metadata in CSV or Parquet, as an "interchange format". In other words, if a an application such as a datapipeline must store intermediate DataFrames to disc then this would be a nice solution.

When saving a DataFrame to CSV, specification of the indexes and dtypes are often lost, e.g. a `RangeIndex` is merely enccoded as the enumerated elements and a `DatetimeIndex` ends up losing the `freq` attribute. When saving in Parquet format with `pyarrow` further restrictions apply: in general, an `Index` can only be stored if the elements are all of the same time. Ad hoc work-arounds can be used but these soon start to become messy.

The other aspect is storing of additional metadata. Even temporary data storage within an application can benefit from associating metainformation with data. This is an easy way to accomplish that.

We do not try to reinvent-the-wheel, only supplement existing functionality found in Pandas.


## Usage

`import df_file_interchange as fi`


