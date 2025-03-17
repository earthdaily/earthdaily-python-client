import unittest

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Polygon

import earthdaily


class TestZonalStats(unittest.TestCase):
    def setUp(self, constant=np.random.randint(1, 12)):
        # Define time, x, and y values
        times = ["1987-04-22", "2022-04-22"]
        x_values = np.arange(0, 8)
        y_values = np.arange(0, 3)
        self.constant = constant
        # Create 3D arrays for the data values
        data_values = np.arange(0, 24).reshape(3, 8)
        data_values = np.dstack((data_values, np.full((3, 8), constant)))

        # Create the xarray dataset
        ds = xr.Dataset(
            {"first_var": (("y", "x", "time"), data_values)},
            coords={
                "y": y_values,
                "x": x_values,
                "time": times,
            },
        ).rio.write_crs("EPSG:4326")
        # first pixel

        geometry = [
            Polygon([(0, 0), (0, 1.2), (1.2, 1.2), (1.2, 0)]),
            Polygon([(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)]),
            Polygon([(1, 1), (9, 1), (9, 2.1), (1, 1)]),
        ]

        gdf = gpd.GeoDataFrame({"geometry": geometry}, crs="EPSG:4326")
        gdf.index = ["ok1", "nok", "ok2"]
        gdf["label"] = [1, 1, 5]
        self.gdf = gdf
        self.datacube = ds

    def test_basic(self):
        zonalstats = earthdaily.earthdatastore.cube_utils.zonal_stats(
            self.datacube,
            self.gdf,
            method="numpy",
            reducers=["min", "max"],
            all_touched=False,
        )
        for operation in ["min", "max"]:
            self._check_results(
                zonalstats["first_var"].sel(zonal_statistics=operation).values,
                operation=operation,
            )

    def _check_results(self, stats_values, operation="min"):
        results = {
            "min": np.asarray([[1, 10], [self.constant, self.constant]]),
            "max": np.asarray([[9, 15], [self.constant, self.constant]]),
        }
        self.assertTrue(np.all(stats_values == results[operation]))

    def test_compare_ed_xvec(self):
        res = {}
        for method in ["numpy", "xvec"]:
            res[method] = earthdaily.earthdatastore.cube_utils.zonal_stats(
                self.datacube,
                self.gdf,
                method="numpy",
                reducers=["max"],
                all_touched=False,
            )
        self.assertTrue(
            np.array_equal(
                res["numpy"]["first_var"].values, res["xvec"]["first_var"].values
            )
        )


if __name__ == "__main__":
    unittest.main()
