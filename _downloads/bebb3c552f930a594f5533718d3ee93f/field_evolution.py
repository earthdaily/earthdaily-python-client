"""
Field evolution and zonal stats
=================================================================

Using Agriculture cloud mask from EarthDaily, and data from L2A, zonal stats for evolution"""

##############################################################################
# Import librairies
# -------------------------------------------


import geopandas as gpd
from matplotlib import pyplot as plt
from earthdaily import datasets, EarthDataStore

##############################################################################
# Load plot
# -------------------------------------------

# load geojson
pivot = datasets.load_pivot()

##############################################################################
# Init earthdatastore with environment variables or default credentials
# ----------------------------------------------------------------------------

eds = EarthDataStore()

##############################################################################
# Search for collection items for June 2022.
# where at least 50% of the field is clear according to the native cloudmask.

pivot_cube = eds.datacube(
    "sentinel-2-l2a",
    intersects=pivot,
    datetime=["2022-06"],
    assets=["red", "green", "blue", "nir"],
    mask_with="native",
    clear_cover=50,
)
pivot_cube.clear_percent.plot.scatter(x="time")

##############################################################################
# Add spectral indices using spyndex from earthdaily accessor
# ------------------------------------------------------------

pivot_cube = pivot_cube.ed.add_indices(['NDVI'])

##############################################################################
# Plots cube with SCL with at least 50% of clear data
# ----------------------------------------------------
pivot_cube = pivot_cube.load()
pivot_cube.ed.plot_rgb(col_wrap=4, vmin=0, vmax=.3)
plt.title("Pivot evolution masked with native cloudmasks")
plt.show()


##############################################################################
# Compute zonal stats for the pivot
# ----------------------------------------------------

zonal_stats = pivot_cube.ed.zonal_stats(pivot, ['mean','max','min'])

zonal_stats.isel(feature=0).to_array(dim="band").plot.line(
    x="time", col="band", hue="zonal_statistics", col_wrap=3
)
plt.show()