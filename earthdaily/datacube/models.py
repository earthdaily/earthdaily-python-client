from enum import Enum
from typing import Literal

ResamplingMethod = Literal["nearest", "bilinear", "cubic", "average", "mode", "gauss", "max", "min", "med", "q1", "q3"]
AggregationMethod = Literal["mean", "median", "min", "max", "sum", "std", "var"]
CompatType = Literal["identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"]
DatacubeEngine = Literal["odc"]


class GroupByOption(str, Enum):
    DATE = "time.date"
