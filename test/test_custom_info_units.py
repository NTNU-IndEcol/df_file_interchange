"""
Tests custom info units
"""

import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import pytest

TESTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(TESTPATH, ".."))

from datetime import datetime, date

from pydantic import (
    ValidationError,
)

import df_file_interchange as fi
from df_file_interchange.ci import unit
from df_file_interchange.ci.unit.base import FIBaseUnit
from df_file_interchange.ci.unit.currency import FICurrencyUnit
from df_file_interchange.ci.unit.population import FIPopulationUnit


def test_unit_currency():

    # Check we get a validation error when trying to set unit_year without
    # unit_year_method
    with pytest.raises(ValidationError):
        currency_unit_year_no_method = FICurrencyUnit(
            unit_desc="USD", unit_multiplier=1.0, unit_year=2004
        )

    # Check we get a validation error when trying to set both unit_year and
    # unit_date
    with pytest.raises(ValidationError):
        currency_unit_year_and_unit_date = FICurrencyUnit(unit_desc="USD", unit_multiplier=1.0, unit_year=2004, unit_year_method="AVG", unit_date="2032-04-23T10:20:30.400+02:30")  # type: ignore
