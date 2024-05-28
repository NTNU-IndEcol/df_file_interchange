# Using Custom Structured Info

There's support for storing both unstructured and custom structured (verifiable) metadata in the metafile. This is under `df_file_interchange.ci` ("custom info"). 

Strictly speaking, the user can pass any object that inherits from `FIBaseCustomInfo` (a Pydantic model), which can fully serialize itself. There is, however, a standard set of these classes already defined to cover most use-cases.

The  `df_file_interchange.ci.structured.FIStructuredCustomInfo` class is the canonical way to proceed. It has two attributes: `extra_info: FIBaseExtraInto` and `col_units: dict[Any, FIBaseUnit]`. The `extra_info` is for stroing metadata that applies to the whole dataframe such as author or its source. The `col_units` is a dictionary that specifies the unit, such as currency, columnwise. There are useful classes for both of these under `df_file_interchange.ci.extra` and `df_file_interchange.ci.unit`, respectively.

## Basic Example

A basic example is included here and also in the corresponding [notebook on using custom info](./notebooks/tutorial_simple_structured_custom_info.ipynb).

```python
import pandas as pd
import numpy as np
import df_file_interchange as fi
from pathlib import Path

from df_file_interchange.ci.extra.std_extra import FIStdExtraInfo
from df_file_interchange.ci.structured import FIStructuredCustomInfo
from df_file_interchange.ci.unit.currency import FICurrencyUnit
from df_file_interchange.ci.unit.population import FIPopulationUnit
```

Next, we create an example dataframe. We also an example extra_info object, some units, and put those together as custom info.

```python
# Create basic dataframe
df = pd.DataFrame(np.random.randn(3, 4), columns=["a", "b", "c", "d"])
df["pop"] = pd.array([1234, 5678, 91011])

# Define some units
unit_cur_a = FICurrencyUnit(unit_desc="USD", unit_multiplier=1000)
unit_cur_b = FICurrencyUnit(unit_desc="EUR", unit_multiplier=1000)
unit_cur_c = FICurrencyUnit(unit_desc="JPY", unit_multiplier=1000000)
unit_cur_d = FICurrencyUnit(unit_desc="USD", unit_multiplier=1000)
unit_pop = FIPopulationUnit(unit_desc="people", unit_multiplier=1)

# Define some extra info
extra_info = FIStdExtraInfo(author="Spud", source="Potato")

# Put that together into a custom_info object
custom_info = FIStructuredCustomInfo(
    extra_info=extra_info,
    col_units={
        "a": unit_cur_a,
        "b": unit_cur_b,
        "c": unit_cur_c,
        "d": unit_cur_d,
        "pop": unit_pop,
    },
)
```

We write it to a file.

```python
data_dir = Path("./data/")
data_dir.mkdir(exist_ok=True)
datafile_csv = Path(data_dir / "tutorial_simple_structured_custom_info.csv")

metafile = fi.write_df_to_file(df, datafile_csv, custom_info=custom_info)
```

Read it in again.

```python
(df_reload, metainfo_reload) = fi.read_df(metafile)
```

And, we can inspect the custom info we have read back in.

```python
metainfo_reload.custom_info
```

```
FIStructuredCustomInfo(unstructured_data={}, extra_info=FIStdExtraInfo(author='Spud', source='Potato', classname='FIStdExtraInfo'), col_units={'a': FICurrencyUnit(unit_desc='USD', unit_multiplier=1000.0, unit_year=None, unit_year_method=None, unit_date=None, classname='FICurrencyUnit'), 'b': FICurrencyUnit(unit_desc='EUR', unit_multiplier=1000.0, unit_year=None, unit_year_method=None, unit_date=None, classname='FICurrencyUnit'), 'c': FICurrencyUnit(unit_desc='JPY', unit_multiplier=1000000.0, unit_year=None, unit_year_method=None, unit_date=None, classname='FICurrencyUnit'), 'd': FICurrencyUnit(unit_desc='USD', unit_multiplier=1000.0, unit_year=None, unit_year_method=None, unit_date=None, classname='FICurrencyUnit'), 'pop': FIPopulationUnit(unit_desc='people', unit_multiplier=1, classname='FIPopulationUnit')}, classname='FIStructuredCustomInfo')
```

(not very readable)

We'll do a string dump (serialization) of the custom info:

```python
metainfo_reload.custom_info.model_dump()
```

```
{'unstructured_data': {},
 'extra_info': {'author': 'Spud',
  'source': 'Potato',
  'classname': 'FIStdExtraInfo'},
 'col_units': {'a': {'unit_desc': 'USD',
   'unit_multiplier': 1000.0,
   'unit_year': None,
   'unit_year_method': None,
   'unit_date': None,
   'classname': 'FICurrencyUnit'},
  'b': {'unit_desc': 'EUR',
   'unit_multiplier': 1000.0,
   'unit_year': None,
   'unit_year_method': None,
   'unit_date': None,
   'classname': 'FICurrencyUnit'},
  'c': {'unit_desc': 'JPY',
   'unit_multiplier': 1000000.0,
   'unit_year': None,
   'unit_year_method': None,
   'unit_date': None,
   'classname': 'FICurrencyUnit'},
  'd': {'unit_desc': 'USD',
   'unit_multiplier': 1000.0,
   'unit_year': None,
   'unit_year_method': None,
   'unit_date': None,
   'classname': 'FICurrencyUnit'},
  'pop': {'unit_desc': 'people',
   'unit_multiplier': 1,
   'classname': 'FIPopulationUnit'}},
 'classname': 'FIStructuredCustomInfo'}
```



## Extending with Your Own Classes

It's easy to extend the provided custom info classes. The important thing is to include a context when reading the dataframe again, so that df_file_interchange knows which classes to instantiate.

Here, we use an example from the tests.

```python
class FITestUnit(FIBaseUnit):
    unit_desc: str = ""
    hamster_type: Literal["syrian", "space-hamster"] = "syrian"


class FITestExtraInfo(FIStdExtraInfo):
    hamster_collective: str | None = None


class FITestStructuredCustomInfo(FIStructuredCustomInfo):
    hamster_motto: str | None = None

    df = pd.DataFrame(
        np.random.randn(4, 3),
        columns=["pet hamsters", "wild hamsters", "space hamsters"],
    )

    unit_cur_pet = FITestUnit(unit_desc="hamster", hamster_type="syrian")
    unit_cur_wild = FITestUnit(unit_desc="hamster", hamster_type="syrian")
    unit_cur_space = FITestUnit(unit_desc="hamster", hamster_type="space-hamster")

    extra_info = FITestExtraInfo(
        author="Spud", source="Potato", hamster_collective="going nuts"
    )

    custom_info = FITestStructuredCustomInfo(
        extra_info=extra_info,
        col_units={
            "unit_cur_pet": unit_cur_pet,
            "unit_cur_wild": unit_cur_wild,
            "unit_cur_space": unit_cur_space,
        },
        hamster_motto="jump, baby",
    )

    # Save CSV
    target_datafile_csv = tmp_path / "test_df_extend_ci_1__csv.csv"
    metafile = fi.write_df_to_csv(
        df,
        target_datafile_csv,
        custom_info=custom_info,
    )

    # Define necessary context
    context = {
        "clss_custom_info": {
            "FITestStructuredCustomInfo": FITestStructuredCustomInfo,
        },
        "clss_extra_info": {
            "FITestExtraInfo": FITestExtraInfo,
        },
        "clss_col_units": {
            "FITestUnit": FITestUnit,
        },
    }

    # Read
    (df_reload, metainfo_reload) = fi.read_df(metafile, context_metainfo=context)

    # Compare dataframes
    chk_strict_frames_eq_ignore_nan(
        df,
        df_reload,
    )

    # Check classes are correct
    assert isinstance(metainfo_reload.custom_info, FITestStructuredCustomInfo)
    assert isinstance(metainfo_reload.custom_info.extra_info, FITestExtraInfo)
    assert isinstance(metainfo_reload.custom_info.col_units["unit_cur_pet"], FITestUnit)
    assert isinstance(
        metainfo_reload.custom_info.col_units["unit_cur_wild"], FITestUnit
    )
    assert isinstance(
        metainfo_reload.custom_info.col_units["unit_cur_space"], FITestUnit
    )

```



