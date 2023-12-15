"""
Field evolution and zonal stats
=================================================================

Using Agriculture cloud mask from EarthDaily, and data from L2A, zonal stats for evolution"""

##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily import earthdatastore, datasets
import geopandas as gpd
from matplotlib import pyplot as plt

##############################################################################
# Load plot
# -------------------------------------------

# load geojson
pivot = datasets.load_pivot()

##############################################################################
# Init earthdatastore with env params
# -------------------------------------------

eds = earthdatastore.Auth()

##############################################################################
# Search for collection items in august 2022 (1st to 9th)
# where at least 50% of the field is clear according to the native cloudmask.

pivot_cube = eds.datacube(
    "sentinel-2-l2a",
    intersects=pivot,
    datetime=["2022-07-01", "2022-08-31"],
    assets=["red", "green", "blue"],
    mask_with="ag_cloud_mask",
    clear_cover=50,
)

pivot_cube.clear_percent.plot.scatter(x="time")

##############################################################################
# Plots cube with SCL with at least 50% of clear data
# ----------------------------------------------------
pivot_cube = pivot_cube.load()
pivot_cube.to_array(dim="band").plot.imshow(
    vmin=0, vmax=0.4, col="time", col_wrap=3
)

plt.title("Clear cover percent with SCL")
plt.title("Pivot evolution with SCL masks")
plt.show()


##############################################################################
# Compute zonal stats for the pivot
# ----------------------------------------------------

zonal_stats = earthdatastore.cube_utils.zonal_stats(
    pivot_cube, pivot, operations=["mean", "max", "min"]
)
zonal_stats = zonal_stats.load()

zonal_stats.isel(feature=0).to_array(dim="band").plot.line(
    x="time", col="band", hue="stats"
)
