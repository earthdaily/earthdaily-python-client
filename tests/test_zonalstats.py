import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import Polygon
import earthdaily
import unittest


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
            Polygon([(0, 0), (0, 0.8), (0.8, 0.8), (0.8, 0)]),
            Polygon([(1, 1), (9, 1), (9, 2.1), (1, 1)]),
        ]
        # out of bound geom #            Polygon([(10,10), (10,11), (11,11), (11,10)])]
        gdf = gpd.GeoDataFrame({"geometry": geometry}, crs="EPSG:4326")
        self.gdf = gdf
        self.datacube = ds

    def test_numpy(self):
        zonalstats = earthdaily.earthdatastore.cube_utils.zonal_stats_numpy(
            self.datacube,
            self.gdf,
            all_touched=True,
            operations=dict(
                mean=np.nanmean, max=np.nanmax, min=np.nanmin, mode=np.mode
            ),
        )

        for operation in ["min", "max", "mode"]:
            self._check_results(
                zonalstats["first_var"].sel(stats=operation).values, operation=operation
            )

    def test_basic(self):
        zonalstats = earthdaily.earthdatastore.cube_utils.zonal_stats(
            self.datacube, self.gdf, all_touched=True, operations=["min", "max", "mode"]
        )
        for operation in ["min", "max", "mode"]:
            self._check_results(
                zonalstats["first_var"].sel(stats=operation).values, operation=operation
            )

    def _check_results(self, stats_values, operation="min"):
        results = {
            "min": np.asarray([[0, self.constant], [9, self.constant]]),
            "max": np.asarray([[8, self.constant], [23, self.constant]]),
            "mode": np.asarray([[0, self.constant], [9, self.constant]]),
        }
        self.assertTrue(np.all(stats_values == results[operation]))


if __name__ == "__main__":
    unittest.main()
