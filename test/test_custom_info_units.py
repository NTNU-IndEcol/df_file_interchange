"""
Tests custom info units
"""

import pytest

from pydantic import (
    ValidationError,
)

from df_file_interchange.ci.unit.currency import FICurrencyUnit


def test_unit_currency():

    # Probably don't want this behaviour any longer...   20240528
    # # Check we get a validation error when trying to set unit_year without
    # # unit_year_method
    # with pytest.raises(ValidationError):
    #     currency_unit_year_no_method = FICurrencyUnit(  # noqa: F841
    #         unit_desc="USD", unit_multiplier=1.0, unit_year=2004
    #     )

    # Check we get a validation error when trying to set both unit_year and
    # unit_date
    with pytest.raises(ValidationError):
        currency_unit_year_and_unit_date = FICurrencyUnit(  # noqa: F841
            unit_desc="USD",
            unit_multiplier=1.0,
            unit_year=2004,
            unit_year_method="AVG",
            unit_date="2032-04-23T10:20:30.400+02:30",  # type: ignore
        )
