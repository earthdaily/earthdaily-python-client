"""
Field evolution and zonal stats
=================================================================

Using SCL data from L2A, zonal stats for evolution"""

##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily import earthdatastore
import geopandas as gpd
from matplotlib import pyplot as plt
import numpy as np

##############################################################################
# Load plot
# -------------------------------------------

# load geojson
pivot = gpd.read_file("pivot.geojson")

##############################################################################
# Init earthdatastore with env params
# -------------------------------------------

eds = earthdatastore.Auth()

##############################################################################
# Search for collection items in august 2022
#
pivot_cube = eds.datacube(
    "sentinel-2-l2a",
    intersects=pivot,
    datetime=["2022-08-01", "2022-08-08"],
    assets=["red", "green", "blue"],
    mask_with="native",  # same as scl
    mask_statistics=True,
)

pivot_cube.clear_percent_scl.plot.scatter(x="time")

#####################################################################da#########
# Plots cube with SCL with at least 50% of clear data
# ----------------------------------------------------
cube_majority_clear = pivot_cube.sel(
    time=pivot_cube.time[pivot_cube.clear_percent_scl > 50]
)

cube_majority_clear = cube_majority_clear.load()

cube_majority_clear.to_array(dim="band").plot.imshow(
    vmin=0, vmax=0.33, col="time", col_wrap=3
)

plt.title("Clear cover percent with SCL")
plt.title("Pivot evolution with SCL masks")
plt.show()


#####################################################################da#########
# Compute zonal stats for the pivot
# ----------------------------------------------------

zonal_stats = earthdatastore.cube_utils.zonal_stats(
    cube_majority_clear, pivot, operations=["mean", "max", "min"]
)
zonal_stats = zonal_stats.load()

zonal_stats.isel(feature=0).to_array(dim="band").plot.line(
    x="time", col="band", hue="stats"
)
