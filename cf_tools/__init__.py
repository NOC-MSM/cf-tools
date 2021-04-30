"""
__init__.py
"""

from pkg_resources import DistributionNotFound, get_distribution

from .accessor import Accessor
from .nemo import NemoAccessor

__all__ = (
    "Accessor",
    "NemoAccessor",
)

try:
    __version__ = get_distribution("cf_tools").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"
