"""
Tests structured custom metainfo handling/storage
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(TESTPATH, ".."))

import df_file_interchange as fi
from df_file_interchange.file.rw import chk_strict_frames_eq_ignore_nan


