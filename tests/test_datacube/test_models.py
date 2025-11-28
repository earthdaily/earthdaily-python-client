import unittest
from typing import get_args

from earthdaily.datacube.models import (
    AggregationMethod,
    CompatType,
    DatacubeEngine,
    ResamplingMethod,
)


class TestModels(unittest.TestCase):
    def test_resampling_method_type(self):
        args = get_args(ResamplingMethod)
        expected_methods = [
            "nearest",
            "bilinear",
            "cubic",
            "average",
            "mode",
            "gauss",
            "max",
            "min",
            "med",
            "q1",
            "q3",
        ]
        for method in expected_methods:
            self.assertIn(method, args)

    def test_resampling_method_valid_values(self):
        valid_values = ["nearest", "bilinear", "cubic", "average", "mode", "gauss", "max", "min", "med", "q1", "q3"]
        for value in valid_values:
            self.assertIn(value, get_args(ResamplingMethod))

    def test_aggregation_method_type(self):
        args = get_args(AggregationMethod)
        expected_methods = ["mean", "median", "min", "max", "sum", "std", "var"]
        for method in expected_methods:
            self.assertIn(method, args)

    def test_aggregation_method_valid_values(self):
        valid_values = ["mean", "median", "min", "max", "sum", "std", "var"]
        for value in valid_values:
            self.assertIn(value, get_args(AggregationMethod))

    def test_compat_type(self):
        args = get_args(CompatType)
        expected_types = ["identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"]
        for compat_type in expected_types:
            self.assertIn(compat_type, args)

    def test_compat_type_valid_values(self):
        valid_values = ["identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"]
        for value in valid_values:
            self.assertIn(value, get_args(CompatType))

    def test_datacube_engine_type(self):
        args = get_args(DatacubeEngine)
        self.assertIn("odc", args)

    def test_datacube_engine_valid_value(self):
        self.assertIn("odc", get_args(DatacubeEngine))


if __name__ == "__main__":
    unittest.main()
