"""
Column unit and description handler




"""

from pprint import pprint
from typing import Any, Literal, TypeAlias, Union

import numpy as np
import pandas as pd
from loguru import logger



from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_serializer,
    model_validator,
)







class FIBaseColumnUnits(BaseModel):

    # The unit, should be a unit_type unit, e.g. if unit_type is "currency" then unit_desc could be "USD"
    unit_desc: Literal["USD"] | None

    # Sometimes we have quantities in "millions of $", for example
    unit_multiplier: float = 1.0


class FICurrencyColumnUnits(BaseModel):

    # Currency abbreviations can be obtained from https://treasury.un.org/operationalrates/OperationalRates.php
    # Download the EXCEL file, copy the column into a text file, then run
    # `cat currency_abbreviations.txt | sed 's/ //g;s/^/"/;s/$/",/' | sort | uniq  > currency_abbreviations_processed.txt`
    # and then you'll need to manually remove the "USDollar" entry.

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
        "CHF",
        "CLP",
        "CNY",
        "CongoFranc",
        "COP",
        "CRC",
        "CUP",
        "CVE",
        "CZK",
        "Denar",
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
        "KoreanWon",
        "KWD",
        "KYD",
        "KZT",
        "LaosKip",
        "LBP",
        "Lilangeni",
        "LKR",
        "LRD",
        "LSL",
        "LYD",
        "MAD",
        "Madagascar",
        "MMK",
        "MNT",
        "MoldovanLeu",
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
        "TanzaniaShilling",
        "THB",
        "TJS",
        "TMT",
        "TND",
        "TOP",
        "TRY",
        "TTD",
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
        "YemeniRial",
        "ZAR",
        "ZMW",
        "ZWL",
    ]

    # Sometimes we have quantities in "millions of $", for example
    unit_multiplier: float = 1.0




class FIColumnSetSpec(BaseModel):

    pass