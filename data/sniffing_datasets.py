"""
Created Sep 25 2022 by Soeren Brandt

This file contains all the sniffing DATASETS used in machine learning
"""

import os
import sqlite3
from abc import ABC
from collections import OrderedDict
from typing import List, Union

import numpy as np
import pandas as pd

from .datasets import Dataset


class AlkanesShortSniff(Dataset):
    def __init__(self):
        """Experiments used in the short sniffing plot for Alkanes."""
        exp_set = OrderedDict(
            [
                ("Pentane", [184, 187, 190, 193, 196, 199, 202, 205]),
                ("Hexane", [183, 186, 189, 192, 195, 198, 201, 204, 207]),
                ("Heptane", [182, 185, 188, 191, 194, 197, 200, 203, 206]),
                ("Octane", [212, 215, 218, 221, 224, 227, 230, 233, 236, 239]),
                ("Nonane", [213, 216, 219, 222, 225, 228, 231, 234, 237, 240]),
                ("Decane", [214, 217, 220, 223, 226, 229, 232, 235, 238, 241]),
            ]
        )
        super().__init__(exp_set)
