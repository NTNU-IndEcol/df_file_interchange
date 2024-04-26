"""
Column currency unit using three letter acronyms, e.g. "USD", "EUR", etc.

"""

from typing import Any, Literal, TypeAlias, Union
from typing_extensions import Self
from loguru import logger

from datetime import datetime, date

from pydantic import (
    BaseModel,
    computed_field,
    model_validator,
    ValidationError,
)

from .base import FIBaseUnit


class FICurrencyUnit(FIBaseUnit):

    # Currency codes can be obtained from https://treasury.un.org/operationalrates/OperationalRates.php
    # Download the EXCEL file, copy the column into a text file, then run
    # `cat currency_abbreviations.txt | sed 's/ //g;s/^/"/;s/$/",/' | sort | uniq  > currency_abbreviations_processed.txt`
    # and then you'll need to manually remove the "USDollar" entry.

    # Supplemented with Taiwan dollar, "TWD"

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
        "TWD", # Manually added
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
    ]

    # Sometimes we have quantities in "millions of $", for example
    unit_multiplier: float = 1.0

    # Sometimes we need currency to be tagged to a specific year, e.g. "EUR" in
    # 2004. If using this field, must also specify whether it's averaged, year
    # end, or what in unit_year_method
    unit_year: int | None = None
    unit_year_method: Literal["AVG", "END"] | None = None

    # Sometimes we might want tosspecify currency against a fixed date.
    unit_date: datetime | date | None = None

    @model_validator(mode="after")
    def model_validator_after(self) -> Self:

        # Check if unit_year not None then unit_year_method must also be not
        # None
        if not self.unit_year is None and self.unit_year_method is None:
            error_msg = f"Validator error: if unit_year is not None then unit_year_method must be defined."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check that both unit_year and unit_date are not both set at once
        if not self.unit_year is None and not self.unit_date is None:
            error_msg = f"Validation error: cannot have both unit_year and unit_date not None at same time."
            logger.error(error_msg)
            raise ValueError(error_msg)

        return self
