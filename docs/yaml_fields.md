# YAML Fields

The YAML file used to store the metainfo contains a number of fields. The data for some of these is serialized using our custom serializer `file.rw._serialize_element()` and `file.rw._deserialize_element()`.

N.B. We store the hash of the datafile in the metainfo, so if you modify the datafile then you also must adjust the hash accordingly. It's an interchange format, so it's assumed users are _NOT_ manually editing stuff.


## General Fields

`datafile`: The filename of the datafile being described.

`encoding`: How the datafile is encoded, is a serialized (just a standard serialization) of an `FIEncoding` object.

`file_format`: Whether datafile is csv or parquet.

`format_version`: We might need to use some sort of versioning for the file format.

`hash`: SHA256 hash of the datafile.


## Index(es)

Both the row and column indexes are encoded in a similar way. Advice: don't mess about trying to edit this manually.

`columns`: Serialized representation of the column index.

`index`: Serialized representation of the row index.


## Dtypes

The column dtypes are specified in `serialized_dtypes`, again in a specific format. Advice: don't try to mess about with this manually.


## Custom Info

Serialized from the structured custom info. Again, in a specific format. Note that the classnames of the relevant Pydantic objects are included here, so that they can be instantiated properly upon reading.


## Example

```yaml
# Metadata for test_with_structured_custom_info.csv
---

columns:
  data:
    el:
    - el: a
      eltype: str
    - el: b
      eltype: str
    - el: c
      eltype: str
    - el: d
      eltype: str
    - el: pop
      eltype: str
    eltype: list
  dtype: object
  index_type: idx
  name: null
custom_info:
  classname: FIStructuredCustomInfo
  col_units:
    a:
      classname: FICurrencyUnit
      unit_date: null
      unit_desc: USD
      unit_multiplier: 1000.0
      unit_year: null
      unit_year_method: null
    b:
      classname: FICurrencyUnit
      unit_date: null
      unit_desc: EUR
      unit_multiplier: 1000.0
      unit_year: null
      unit_year_method: null
    c:
      classname: FICurrencyUnit
      unit_date: null
      unit_desc: JPY
      unit_multiplier: 1000000.0
      unit_year: null
      unit_year_method: null
    d:
      classname: FICurrencyUnit
      unit_date: null
      unit_desc: USD
      unit_multiplier: 1000.0
      unit_year: null
      unit_year_method: null
    pop:
      classname: FIPopulationUnit
      unit_desc: people
      unit_multiplier: 1
  extra_info:
    author: Spud
    classname: FIStdExtraInfo
    source: A simple test
  unstructured_data: {}
datafile: test_with_structured_custom_info.csv
encoding:
  auto_convert_int_to_intna: true
  csv:
    csv_allowed_na:
    - <NA>
    doublequote: true
    float_precision: round_trip
    keep_default_na: false
    na_rep: <NA>
    quoting: 2
    sep: ','
  parq:
    engine: pyarrow
    index: null
file_format: csv
format_version: 1
hash: ac3f2e11e0ff9ba540b630f03d32652560ef6a224778184b30271ce542b4a59a
index:
  dtype: int64
  index_type: range
  name: null
  start: 0
  step: 1
  stop: 3
serialized_dtypes:
  a:
    dtype_str: float64
    serialized_col_name:
      el: a
      eltype: str
  b:
    dtype_str: float64
    serialized_col_name:
      el: b
      eltype: str
  c:
    dtype_str: float64
    serialized_col_name:
      el: c
      eltype: str
  d:
    dtype_str: float64
    serialized_col_name:
      el: d
      eltype: str
  pop:
    dtype_str: Int64
    serialized_col_name:
      el: pop
      eltype: str

```



