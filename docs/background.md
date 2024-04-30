## Technical Background

### Storing Index and Columns (also an Index) in the YAML File

This all sounds a bit dubious at first. However, consider that Pandas has several types of `Index` including `Index`, `RangeIndex`, `DatetimeIndex`, `MultiIndex`, `CategoricalIndex`, etc. Some of these, such as `Index`, represent the index explicitly with list(s) of elements. Others represent the index in a shorthand way, using only a few parameters needed to reconstruct the index, e.g. `RangeIndex`. The former could fit nicely as an additional column or row in the tabluar data but the latter cannot and is better stored in the YAML file.

Ok, so we could, and may eventually, do just that but it adds complexity to the code. Also, the `columns` in Pandas act as the unique identifier for the columns and, unfortuantely, the columns need not be a simple list of str or ints: it can be a `MultiIndex` or such, i.e. something that requires deserialization and instantiation (this has to happen before applying dtypes, for example). There are also further complications in how Pandas handles some of this internally in the sense that it's not entirely consistent.

This arrangement is not ideal but, for now at least, storing the serialized row index and column index in the YAML file seems a reasonably "clean" way to resolve the problem even if this means a much bigger file. We're not storing massive DataFrames so this should be fine since the files are written+read programatically.
