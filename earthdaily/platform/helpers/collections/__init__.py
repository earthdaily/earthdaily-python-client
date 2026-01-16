"""
Collection helpers for EarthDaily platform.

This module provides collection-specific helper classes that make it easier
to work with specific satellite data collections on the EarthDaily platform.
"""

from earthdaily.platform.helpers.collections.sentinel1 import Sentinel1CollectionHelper
from earthdaily.platform.helpers.collections.sentinel2 import Sentinel2CollectionHelper

__all__ = ["Sentinel1CollectionHelper", "Sentinel2CollectionHelper"]
