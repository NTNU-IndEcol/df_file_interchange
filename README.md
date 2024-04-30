# df_file_interchange

For data pipelines, it's often advantageous to be able to "round trip" a dataframe to disc: the results of what is proccessed at one stage being required by independent stages afterwards. Pandas can store and manipulate various structures on top of a basic table including indexes and dtype specifications. These indexes can be generated from a small number of parameters that cannot naturally be encoded in a column, e.g. a `pd.RangeIndex`. When saving to a CSV file, these indexes are typically enumerated. Therefore, when reading back in the dataframe it is not actually the same as the original. Although this is handled better when writing to Parquet format, there are still a few issues such as all entries in a column, and so an index, being required to have the same dtype. Similarly, it's often desirable to store some custom structured metadata along with the dataframe, e.g. the author or columnwise metadata such as what unit each column is expressed in.

`df_file_interchange` is a wrapper around Pandas that tries to address these requirements. It stores additional metadata that (mostly) ensures the indexes are recreated properly, column dtypes are specified properly, and that the dataframe read from disc is actually the same as what was written. It also manages the storage of additional custom metadata.

In the future, the intention is also to opportunistically serialise columns when required since there are some dtypes that are not supported by Parquet.

This is not trying to reinvent-the-wheel, only supplement existing functionality found in Pandas.


## Usage

### Basic Usage for Writing+Reading CSV/Parquet

```python
import df_file_interchange as fi
```

Then do something like (autodetect of target file format from `datafile_path` extension):

```python
metafile = fi.write_df_to_file(df, datafile_path)
```

or to specify the datafile format explicitly:

```python
metafile = fi.write_df_to_csv(df, datafile_path)
```

```python
metafile = fi.write_df_to_parquet(df, datafile_path)
```

where `metafile` will return a `Path` object for the YAML file path, `datafile_path` is a `Path` object.

Additional parameters can be used, e.g. `metafile` to specify the YAML file manually (must be same dir) and `custom_info`.

```python
metafile = fi.write_df_to_file(
    df, datafile_path, metafile_path, "csv"
)
```

Additional encoding options can be specified using the `encoding` argument (as a `FIEncoding` object) but this is unnecessary and probably unwise.

To read:

```python
(df, metainfo) = fi.read_df(yamlfile_path)
```

the `df` is of course the dataframe and `metainfo` is a Pydantic object containing all the metainfo associated with the file encoding, indexes, dtypes, etc, and also the user-supplied custom info.



### Structured Metadata

There's support for storing custom metadata, in the sense that it both can be validated using Pydantic models and is extensible.

_NOTE_: for development purposes, we currently use a "loose" validation, which may result in 'missing' info/units, and a warning; later, this will be made strict and so would raise an exception.

In principle, we can store any custom information that is derived from the `FIBaseCustomInfo` class. However, it's strongly advised to use `FIStructuredCustomInfo`. At time of writing, this supports general 'table-wide' information (as `FIBaseExtraInfo` or `FIStdExtraInfo`), and columnwise units (derived from `FIBaseUnit`).

To allow both validation using Pydantic and extensibility is slightly fiddly: when reading, we have to know which classes to instantiate. By default, the code at the moment checks `globals()` and checks that the supplied class is a subclass of the appropriate base class. However, one can manually define a context to enforce which classes can be instantiated, which might be necessary when extending functionality.

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
extra_info = fi.ci.structured.extra.FIStdExtraInfo(author="Spud", source="Potato")

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
(df_reload, metainfo_reload) = fi.read_df(metafile)
```

If you were extending the extra info or units, you'll probably have to include a context. TODO: document this.


## Known Problems

* Pyarrow won't encode numpy's complex64. So, we've disabled this in the tests for now although the functionality will work in CSV. Solution would be to serialize the actual data column when necessary but that's new functionality.

* Float16s won't work properly and will never work, due to Pandas decision.

* Some of the newer or more esoteric features of Pandas are not yet accounted for.



