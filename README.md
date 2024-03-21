# df_file_interchange

This package is designed to store complete specification DataFrames (including indexes) along with custom metadata in CSV or Parquet, as an "interchange format". In other words, if a an application such as a datapipeline must store intermediate DataFrames to disc then this would be a nice solution.

When saving a DataFrame to CSV, specification of the indexes and dtypes are often lost, e.g. a `RangeIndex` is merely enccoded as the enumerated elements and a `DatetimeIndex` ends up losing the `freq` attribute. When saving in Parquet format with `pyarrow` further restrictions apply: in general, an `Index` can only be stored if the elements are all of the same type. Ad hoc work-arounds can be used but these soon start to become messy.

The other aspect is storing of additional metadata. Even temporary data storage within an application can benefit from associating metainformation with data. This is an easy way to accomplish that.

We do not try to reinvent-the-wheel, only supplement existing functionality found in Pandas.


## Usage

`import df_file_interchange as fi`

Then do something like:

(for CSV)

`metafile = fi.write_df_to_fi_generic(df, datafile_path, yamlfile_path, fi.FIFileFormatEnum.csv, custom_info_dict=custom_info_dict)`

(for Parquet)

`metafile = fi.write_df_to_fi_generic(df, datafile_path, yamlfile_path, fi.FIFileFormatEnum.parquet, custom_info_dict=custom_info_dict)`

where `metafile` will return a `Path` object that is just `yamlfile_path`.

TODO: allow literal specification in place of `fi.FIFileFormatEnum`.

To read:

`(df, metainfo) = fi.read_fi_to_df_generic(yamlfile_path)`

the `df` is of course the dataframe and `metainfo` is a Pydantic object containing all the metainfo associated with the file encoding, indexes, dtypes, etc, and also the user-supplied custom info.


## Known Problems

* Pyarrow won't encode numpy's complex64. So, we've disabled this in the tests for now although the functionality will work in CSV. Solution would be to serialize the actual data column when necessary but that's new functionality.



## Technical Reasonings

### Storing Index and Columns (also an Index) in the YAML File

This sounds unwise at first. However, consider that Pandas has several types of `Index` including `Index`, `RangeIndex`, `DatetimeIndex`, `MultiIndex`, `CategoricalIndex`, etc. Some of these, such as `Index`, represent the index explicitly with list(s) of elements. Others represent the index in a shorthand way, using only a few parameters needed to reconstruct the index, e.g. `RangeIndex`. The former could fit nicely as an additional column or row in the tabluar data but the latter cannot and is better stored in the YAML file.

Ok, so we could, and may eventually, do just that but it adds complexity to the code. Also, the `columns` function as the unique identifier for the columns and, unfortuantely, the columns need not be a simple list of str or ints: it can be a `MultiIndex` or such, i.e. something that requires deserialization and instantiation (this has to happen before applying dtypes, for example).

It's not ideal but, for now at least, storing the serialized row index and column index in the YAML file seems a reasonably "clean" way to resolve the problem even if this means a much bigger file. We're not storing massive DataFrames --at most about 10k x 10k-- so this should be fine since the files are written+read programatically.


