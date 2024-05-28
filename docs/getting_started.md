# Getting Started

## Installation

Install from PyPi or conda-forge.

## Simple Write and Read

Ok, lets first import NumPy, Pandas, pathlib, and df_file_interchange.

There's a copy of this tutorial in the handy [notebook on simple write and write](./notebooks/tutorial_simple_write_read.ipynb).

```python
import numpy as np
import pandas as pd
import df_file_interchange as fi
from pathlib import Path
```

Create a simple dataframe.

```python
df = pd.DataFrame(
    {
        "a": [1, 2, 3, 4, 5],
        "b": ["apples", "pears", "oranges", "bananas", "bears"],
        "c": [np.pi, 2*np.pi, 3*np.pi, 4*np.pi, 5*np.pi],
        "d": [
            np.datetime64("2010-01-31T10:23:01"),
            np.datetime64("2014-01-01T10:23:01"),
            np.datetime64("2018-02-28T10:23:01"),
            np.datetime64("2024-01-31T10:23:01"),
            np.datetime64("1999-01-31T23:59:59")]
    },
    index=pd.RangeIndex(start=10, stop=15, step=1),
)
```

Now, lets save the dataframe as a CSV. Minimally, only the dataframe and the filename of the data file to write to is required. `df_file_interchange` can determine the file format from the extension of the supplied data filename.

```python
data_dir = Path("./data/")
data_dir.mkdir(exist_ok=True)
datafile = Path(data_dir / "tutorial_trying_out_a_save.csv")

metafile = fi.write_df_to_file(df, datafile)
```

This will create two files: `tutorial_trying_out_a_save.csv` and `tutorial_trying_out_a_save.yaml`. The YAML file describes how the CSV file was encoded, the indexes, the dtypes, ancillary data like the hash of the data file, and (more generally) any custom metadata provided.

The `metafile` variable returned is a `Path` object pointing to the location of the YAML file.

Lets read that file back in.

```python
(df_reload, metainfo_reload) = fi.read_df(metafile)
```
The `df_reload` variable is the dataframe as read in from the file. The `metainfo_reload` is a `FIMetaInfo` object.

We can confirm that `df_reload` is the same as the original `df` by running

```python
fi.chk_strict_frames_eq_ignore_nan(df, df_reload)
```

This is just a wrapper around `pd._testing.assert_frame_equal()` where we instead assume `NaN == NaN` (unlike the standards definition that `NaN != NaN`).

## Variations on a Theme

For writing a df to file, there are convenience functions that specify the output type (`fi.write_df_to_parquet()` and `fi.write_df_to_csv()`).

```python
datafile_parq = Path(data_dir / "tutorial_trying_out_a_save.parq")
fi.write_df_to_parquet(df, datafile_parq)
```

Additional parameters can be used, e.g. `metafile` to specify the YAML file manually (must be same dir) and `custom_info`.

```python
metafile = fi.write_df_to_file(
    df, datafile, metafile, "csv"
)
```

Additional encoding options can be specified using the `encoding` argument (as a `FIEncoding` object) but this is unnecessary and probably unwise.

