"""
Create a multisensor cube
=================================================================

With Sentinel-2 and Landsat, using Sentinel-2 spatial resolution."""


##############################################################################
# Import librairies
# -------------------------------------------

from matplotlib import pyplot as plt

from earthdaily import EarthDataStore, datasets

##############################################################################
# Set parameters
# -------------------------------------------

##############################################################################
# Init earthdatastore with environment variables or default credentials
# ----------------------------------------------------------------------------

eds = EarthDataStore()

collections = ["sentinel-2-l2a", "landsat-c2l2-sr"]
datetime = ["2022-07-20", "2022-09-01"]
intersects = datasets.load_pivot_corumba()
assets = ["blue", "green", "red", "nir"]
mask_with = "native"
clear_cover = 50
resampling = "cubic"
cross_calibration_collection = "sentinel-2-l2a"

##############################################################################
# Create the multisensors datacube
# -------------------------------------------

datacube = eds.datacube(
    collections,
    assets=assets,
    datetime=datetime,
    intersects=intersects,
    mask_with=mask_with,
    clear_cover=clear_cover,
    cross_calibration_collection=cross_calibration_collection,
)

# Add the NDVI
datacube = datacube.ed.add_indices(["NDVI"])

# Load in memory
datacube = datacube.load()

##############################################################################
# See the evolution in RGB
# -------------------------------------------

datacube.ed.plot_rgb(col_wrap=3)
plt.show()

##############################################################################
# See the NDVI evolution
# -------------------------------------------

datacube["NDVI"].ed.plot_band(col_wrap=3, vmin=0, vmax=0.8, cmap="Greens")
plt.show()

##############################################################################
# See the NDVI mean evolution
# -------------------------------------------

datacube["NDVI"].groupby("time").mean(...).plot.line(x="time")
plt.title("NDVI evolution")
plt.show()
