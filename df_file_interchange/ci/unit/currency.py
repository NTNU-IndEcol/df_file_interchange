"""
Column currency unit using three letter acronyms, e.g. "USD", "EUR", etc.

"""

from typing import Literal
from typing_extensions import Self
from loguru import logger

from datetime import datetime, date

from pydantic import (
    model_validator,
)

from .base import FIBaseUnit


class FICurrencyUnit(FIBaseUnit):
    """(Pydantic) class that defines currencies as a unit

    Notes
    -----

    For defining these (dev), currency codes can be obtained from
    https://treasury.un.org/operationalrates/OperationalRates.php Download the
    EXCEL file, copy the column into a text file, then run
    `cat currency_abbreviations.txt | sed 's/ //g;s/^/"/;s/$/",/' | sort | uniq  > currency_abbreviations_processed.txt`
    and then you'll need to manually remove the "USDollar" entry.

    Supplemented with Taiwan dollar, "TWD"

    Attributes
    ----------
    unit_desc : Literal
        What currency, e.g. "USD, "EUR", etc.

    unit_multiplier : float
        How many of `unit_desc` at one time, e.g. 1_000_000 USD. Default 1.0.

    unit_year : int or None
        Sometimes we want the currency to be pegged to a specific year, e.g.
        "EUR" in 2004. If using this attribute, must also specify whether its
        average, year end, etc. Default None.
    unit_year_method: Literal["AVG", "END"] | None
        Whether currency has been calculated as an average over a period or is
        end of period. Default None.

    unit_date: datetime | date | None
        Sometimes we might want to specify currency against a fixed date.
        Default None.
    """

    # The various currencies we can use.
    unit_desc: Literal[
        "AED",
        "AFN",
        "ALL",
        "AMD",
        "ANG",
        "AOA",
        "ARS",
        "AUD",
        "AWG",
        "AZN",
        "BAM",
        "BBD",
        "BDT",
        "BGN",
        "BHD",
        "BIF",
        "BMD",
        "BND",
        "BOB",
        "BRL",
        "BSD",
        "BTN",
        "BWP",
        "BYN",
        "BZD",
        "CAD",
        "CDF",
        "CHF",
        "CLP",
        "CNY",
        "COP",
        "CRC",
        "CUP",
        "CVE",
        "CZK",
        "DJF",
        "DKK",
        "DOP",
        "DZD",
        "EGP",
        "ERN",
        "ETB",
        "EUR",
        "FJD",
        "GBP",
        "GEL",
        "GHS",
        "GIP",
        "GMD",
        "GNF",
        "GTQ",
        "GYD",
        "HKD",
        "HNL",
        "HTG",
        "HUF",
        "IDR",
        "ILS",
        "INR",
        "IQD",
        "IRR",
        "ISK",
        "JMD",
        "JOD",
        "JPY",
        "KES",
        "KGS",
        "KHR",
        "KMF",
        "KPW",
        "KRW",
        "KWD",
        "KYD",
        "KZT",
        "LAK",
        "LBP",
        "LKR",
        "LRD",
        "LSL",
        "LYD",
        "MAD",
        "MDL",
        "MGA",
        "MKD",
        "MMK",
        "MNT",
        "MOP",
        "MRU",
        "MUR",
        "MVR",
        "MWK",
        "MXN",
        "MYR",
        "MZN",
        "NAD",
        "NGN",
        "NIO",
        "NOK",
        "NPR",
        "NZD",
        "OMR",
        "PAB",
        "PEN",
        "PGK",
        "PHP",
        "PKR",
        "PLN",
        "PYG",
        "QAR",
        "RON",
        "RSD",
        "RUB",
        "RWF",
        "SAR",
        "SBD",
        "SCR",
        "SDG",
        "SEK",
        "SGD",
        "SHP",
        "SLE",
        "SOS",
        "SRD",
        "SSP",
        "STN",
        "SYP",
        "SZL",
        "THB",
        "TJS",
        "TMT",
        "TND",
        "TOP",
        "TRY",
        "TTD",
        "TWD",  # Manually added
        "TZS",
        "UAH",
        "UGX",
        "USD",
        "UYU",
        "UZS",
        "VES",
        "VND",
        "VUV",
        "WST",
        "XAF",
        "XCD",
        "XOF",
        "XPF",
        "YER",
        "ZAR",
        "ZMW",
        "ZWL",
    ] = "USD"

    # Sometimes we have quantities in "millions of $", for example
    unit_multiplier: float = 1.0

    # Sometimes we need currency to be tagged to a specific year, e.g. "EUR" in
    # 2004. If using this field, can also specify whether it's averaged, year
    # end, etc, in unit_year_method
    unit_year: int | None = None
    unit_year_method: Literal["AVG", "END"] | None = None

    # Sometimes we might want to specify currency against a fixed date.
    unit_date: datetime | date | None = None

    @model_validator(mode="after")
    def model_validator_after(self) -> Self:
        # 20240522: this doesn't necessarily make sense, so have commented it out...
        # # Check if unit_year not None then unit_year_method must also be not
        # # None
        # if self.unit_year is not None and self.unit_year_method is None:
        #     error_msg = "Validator error: if unit_year is not None then unit_year_method must be defined."
        #     logger.error(error_msg)
        #     raise ValueError(error_msg)

        # Check that if unit_year_method not None then unit_year is not None
        if self.unit_year_method is not None and self.unit_year is None:
            error_msg = "Validation error: if unit_year_method not None then unit_year must also be not None."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check unit_year and unit_date aren't both set at the same time
        if self.unit_year is not None and self.unit_date is not None:
            error_msg = "Validation error: unit_year and unit_date cannot both be not None at the same time"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return self
