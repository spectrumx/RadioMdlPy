"""
RadioMdlPy - A Python package for radio modeling calculations
"""

from .RadioMdl import k_boltz, rad, speed_c
from . import antenna_pattern
from . import astro_mdl
from . import coord_frames
from . import obs_mdl
from . import radio_io
from . import radio_types
from . import sat_mdl

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "k_boltz",
    "rad",
    "speed_c",
    "antenna_pattern",
    "astro_mdl",
    "coord_frames",
    "obs_mdl",
    "radio_io",
    "radio_types",
    "sat_mdl",
]