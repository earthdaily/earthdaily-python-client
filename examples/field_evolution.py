"""
Field evolution and zonal stats
=================================================================

Using Agriculture cloud mask from EarthDaily, and data from L2A, zonal stats for evolution"""

##############################################################################
# Import librairies
# -------------------------------------------

from matplotlib import pyplot as plt

import earthdaily as ed

##############################################################################
# Load plot
# -------------------------------------------

# load geojson
pivot = ed.datasets.load_pivot()

##############################################################################
# Init earthdatastore with environment variables or default credentials
# ----------------------------------------------------------------------------

eds = ed.EarthDataStore()

##############################################################################
# Search for collection items for June 2022.
# where at least 50% of the field is clear according to the native cloudmask.

datacube = eds.datacube(
    "sentinel-2-l2a",
    intersects=pivot,
    datetime=["2022-06", "2022-07"],
    assets=["red", "green", "blue", "nir"],
    mask_with="native",
    clear_cover=50,
)
datacube.clear_percent.plot.scatter(x="time")

##############################################################################
# Add spectral indices using spyndex from earthdaily accessor
# ------------------------------------------------------------

datacube = datacube.ed.add_indices(["NDVI"])

##############################################################################
# Plots cube with SCL with at least 50% of clear data
# ----------------------------------------------------
datacube = datacube.load()
datacube.ed.plot_rgb(col_wrap=4, vmin=0, vmax=0.3)
plt.title("Pivot evolution masked with native cloudmasks")
plt.show()

##############################################################################
# Compute zonal stats for the pivot
# ----------------------------------------------------

zonal_stats = datacube.ed.zonal_stats(pivot, ["mean", "max", "min"])
zonal_stats.isel(feature=0).to_array(dim="band").plot.line(
    x="time", col="band", hue="zonal_statistics", col_wrap=3
)
plt.show()
