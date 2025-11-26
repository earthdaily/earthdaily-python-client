"""
Constants and default values for the datacube module.

This module defines all configuration defaults, dimension names, and algorithm
constants used throughout the datacube functionality. Centralizing these values
ensures consistency and makes the codebase easier to maintain.
"""

from earthdaily.datacube.models import AggregationMethod

# Datacube service defaults
DEFAULT_DTYPE = "float32"
DEFAULT_ENGINE = "odc"
DEFAULT_HREF_PATH = "alternate.download.href"
DEFAULT_BBOX_CRS = "EPSG:4326"
DEFAULT_CHUNKS = {"x": "auto", "y": "auto", "time": 1}
DEFAULT_NODATA: float | int | None = None

# Shared dimension names
DIM_TIME = "time"
DIM_X = "x"
DIM_Y = "y"
DIM_LATITUDE = "latitude"
DIM_LONGITUDE = "longitude"
DIM_FEATURE = "feature"
DIM_ZONAL_STATS = "zonal_statistics"
DIM_BANDS = "bands"

# Temporal/aggregation defaults
DEFAULT_WHITTAKER_BETA = 10000.0
DEFAULT_WHITTAKER_FREQ = "1D"
DEFAULT_TEMPORAL_FREQ = "1ME"  # 1 Month End (pandas frequency)
DEFAULT_AGGREGATION: AggregationMethod = "mean"
DEFAULT_ZONAL_REDUCERS = ["mean"]

# Visualization defaults
DEFAULT_RGB_RED = "red"
DEFAULT_RGB_GREEN = "green"
DEFAULT_RGB_BLUE = "blue"
DEFAULT_COLORMAP = "Greys"
DEFAULT_COL_WRAP = 5
DEFAULT_TIME_INDEX = 0

# Whittaker smoothing internals
WHITTAKER_ALPHA = 3
WHITTAKER_BAND_SOLVE = (3, 3)

# Percentage bounds for masking stats
PERCENT_MIN = 0
PERCENT_MAX = 100
