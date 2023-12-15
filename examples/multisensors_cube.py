"""
Create a multisensor cube
=================================================================

With Sentinel-2 and Landsat, using Sentinel-2 spatial resolution."""


##############################################################################
# Import librairies
# -------------------------------------------

import geopandas as gpd
from matplotlib import pyplot as plt

from earthdaily import earthdatastore, datasets

##############################################################################
# Set parameters
# -------------------------------------------

eds = earthdatastore.Auth()
collections = ["sentinel-2-l2a", "landsat-c2l2-sr"]
datetime = ["2022-07-01", "2022-09-01"]
intersects = datasets.load_pivot_corumba()
assets = ["blue", "green", "red", "nir"]
mask_with = "ag_cloud_mask"
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
datacube["ndvi"] = (datacube["nir"] - datacube["red"]) / (
    datacube["nir"] + datacube["red"]
)

# Load in memory
datacube = datacube.load()

##############################################################################
# See the evolution in RGB
# -------------------------------------------

datacube[["red", "green", "blue"]].to_array(dim="band").plot.imshow(
    col="time", col_wrap=3, vmax=0.2
)
plt.show()

##############################################################################
# See the NDVI evolution
# -------------------------------------------

datacube["ndvi"].plot.imshow(
    col="time", col_wrap=3, vmin=0, vmax=0.8, cmap="RdYlGn"
)
plt.show()

##############################################################################
# See the NDVI mean evolution
# -------------------------------------------

datacube["ndvi"].groupby("time").mean(...).plot.line(x="time")
plt.title("NDVI evolution")
plt.show()
