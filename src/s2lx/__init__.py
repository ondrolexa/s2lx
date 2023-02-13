# -*- coding: utf-8 -*-
"""s2lx - simple Sentinel-2 tools

This module contains basic classes to extract datasets from Sentinel-2 SAFE
format suitable for analysis and visualization,

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

"""

from s2lx.s2classes import S2, SAFE, Band, Composite
from s2lx.s2filters import *

__version__ = "0.1.0"
__author__ = "Ondrej Lexa"
__email__ = "lexa.ondrej@gmail.com"
