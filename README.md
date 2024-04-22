# df_file_interchange

This package is designed to store complete specification DataFrames (including indexes) along with structured custom metadata in CSV or Parquet, as an "interchange format". In other words, if a an application such as a datapipeline must store intermediate DataFrames to disc then this would be a nice solution.

When saving a DataFrame to CSV, specification of the indexes and dtypes are often lost, e.g. a `RangeIndex` is merely encoded as the enumerated elements, a `DatetimeIndex` ends up losing the `freq` attribute, etc. When saving in Parquet format with `pyarrow` some restrictions apply: in general, an `Index` can only be stored if the elements are all of the same type. Ad hoc work-arounds can be used but these soon start to become messy.

The other aspect is storing of additional metadata. Even temporary tabular data storage within an application can benefit from having associated metadata. The aim is that this is done in a structured manner and that it's extensible, in the sense that some simple examples are included here but the user can easily define their own custom metadata structure. So, for example, in a economics setting, one may wish to denoate some columns as being a currency (USD, EUR, etc) with multiplier (millions of EUR, say). In a different setting, one might want physical units, say.

We do not try to reinvent-the-wheel, only supplement existing functionality found in Pandas.


## Usage

### Basic Usage for Writing+Reading CSV/Parquet

`import df_file_interchange as fi`

Then do something like (autodetect of target file format from `datafile_path` extension):

`metafile = fi.write_df_to_file(df, datafile_path, yamlfile_path, custom_info_dict=custom_info_dict)`

or to specify the datafile format explicitly:

`metafile = fi.write_df_to_csv(df, datafile_path, custom_info=custom_info)`

`metafile = fi.write_df_to_parquet(df, datafile_path, custom_info=custom_info)`

where `metafile` will return a `Path` object that is just `yamlfile_path`, `datafile_path` and `yamlfile_path` are `Path` objects, and `custom_info` is a dictionary (custom_info will change slightly once the structured metadata code is finished).

To read:

`(df, metainfo) = fi.read_df(yamlfile_path)`

the `df` is of course the dataframe and `metainfo` is a Pydantic object containing all the metainfo associated with the file encoding, indexes, dtypes, etc, and also the user-supplied custom info.

Additional encoding options can be specified using the `encoding` argument (as a `FIEncoding` object) but this is unnecessary and probably unwise.


### Structured Metadata

There's support for storing custom metadata, in the sense that it both can be validated using Pydantic models and is extensible. _NOTE_: for development purposes, there is currently a "loose" validation, which may result in 'missing' info/units, and a warning; later, this will be made strict and so would raise an exception.

In principle, we can store any custom information that is derived from the `FIBaseCustomInfo` class. However, it's strongly advised to use `FIStructuredCustomInfo`. At time of writing, this supports general 'table-wide' information (as `FIBaseExtraInfo` or `FIStdExtraInfo`), and columnwise units (derived from `FIBaseUnit`). To allow both validation using Pydantic and extensibility is slightly fiddly: when reading, we have to know which classes to instantiate. The code to do this is included in the aforementioned classes, so it should be simple enough to derive from these to maintain that functionality.

This means that, when reading, a "context" must be supplied. In short, this is information on what custom info classes are available. For anything included in df_file_interchange though, this can be done automatically. It's only if you're extending that you have to concern yourself with this.

Anyway, a simple example (same example used in tests).

```python
# Create simple dataframe
df = pd.DataFrame(np.random.randn(3, 4), columns=["a", "b", "c", "d"])
df["pop"] = pd.array([1234, 5678, 101010])

# Create units
unit_cur_a = fi.ci.unit.currency.FICurrencyUnit(unit_desc="USD", unit_multiplier=1000)
unit_cur_b = fi.ci.unit.currency.FICurrencyUnit(unit_desc="EUR", unit_multiplier=1000)
unit_cur_c = fi.ci.unit.currency.FICurrencyUnit(unit_desc="JPY", unit_multiplier=1000000)
unit_cur_d = fi.ci.unit.currency.FICurrencyUnit(unit_desc="USD", unit_multiplier=1000)
unit_pop = fi.ci.unit.population.FIPopulationUnit(unit_desc="people", unit_multiplier=1)

# Create extra info
extra_info = fi.ci.structured.FIStdExtraInfo(author="Spud", source="A simple test")

# Create custom info with these
custom_info = fi.ci.structured.FIStructuredCustomInfo(
    extra_info=extra_info,
    col_units={
        "a": unit_cur_a,
        "b": unit_cur_b,
        "c": unit_cur_c,
        "d": unit_cur_d,
        "pop": unit_pop,
    },
)

# Write to file
metafile = fi.write_df_to_csv(df, Path("./test_with_structured_custom_info.csv"), custom_info=custom_info)
```

and to read (we use the default context)

```python
(df_reload, metainfo_reload) = fi.read_df(metafile, context_metainfo=fi.generate_default_context())
```

If you were extending the extra info or units, you'd need to include extra information in the context. TODO: document this.


## Known Problems

* Pyarrow won't encode numpy's complex64. So, we've disabled this in the tests for now although the functionality will work in CSV. Solution would be to serialize the actual data column when necessary but that's new functionality.



## Technical Reasonings

### Storing Index and Columns (also an Index) in the YAML File

This all sounds a bit dubious at first. However, consider that Pandas has several types of `Index` including `Index`, `RangeIndex`, `DatetimeIndex`, `MultiIndex`, `CategoricalIndex`, etc. Some of these, such as `Index`, represent the index explicitly with list(s) of elements. Others represent the index in a shorthand way, using only a few parameters needed to reconstruct the index, e.g. `RangeIndex`. The former could fit nicely as an additional column or row in the tabluar data but the latter cannot and is better stored in the YAML file.

Ok, so we could, and may eventually, do just that but it adds complexity to the code. Also, the `columns` in Pandas act as the unique identifier for the columns and, unfortuantely, the columns need not be a simple list of str or ints: it can be a `MultiIndex` or such, i.e. something that requires deserialization and instantiation (this has to happen before applying dtypes, for example). There are also further complications in how Pandas handles some of this internally in the sense that it's not entirely consistent.

This arrangement is not ideal but, for now at least, storing the serialized row index and column index in the YAML file seems a reasonably "clean" way to resolve the problem even if this means a much bigger file. We're not storing massive DataFrames so this should be fine since the files are written+read programatically.


