"""
Create a multisensor cube
=================================================================

With Sentinel-2 and Venus, using Sentinel-2 spatial resolutino for demo purpose"""


##############################################################################
# Import librairies
# -------------------------------------------

import geopandas as gpd
from matplotlib import pyplot as plt
from rasterio.enums import Resampling

from earthdaily import earthdatastore, datasets

##############################################################################
# Import librairies
# -------------------------------------------

eds = earthdatastore.Auth()
polygon = datasets.load_pivot()

datetime = ["2022-08"]


##############################################################################
# Generate s2 cube
# -------------------------------------------

assets = ["blue", "green", "red", "nir"]

s2 = eds.datacube(
    "sentinel-2-l2a",
    intersects=polygon,
    datetime=datetime,
    assets=assets,
)

##############################################################################
# Generate venus cube
# -------------------------------------------

landsat = eds.datacube(
    "landsat-c2l2-sr",
    intersects=polygon,
    resolution=s2.rio.resolution()[0],
    datetime=datetime,
    epsg=s2.rio.crs.to_epsg(),
    resampling=Resampling.nearest,  # cubic
    assets=assets,
    cross_calibration_collection="sentinel-2-l2a" # cross calibrate using EarthDaily Agro coefficient
)


##############################################################################
# Create supercube
# -------------------------------------------

print("create metacube")
supercube = earthdatastore.metacube(s2, landsat)

##############################################################################
# Get the first common date between S2 and Venus for plotting
# ---------------------------------------------------------------

common_date = [
    day
    for day in s2.time.dt.strftime("%Y%m%d").values
    if day in landsat.time.dt.strftime("%Y%m%d").values
][0]

##############################################################################
# Plot sentinel-2
# -------------------------------------------

s2.sel(time=common_date)[["red", "green", "blue"]].to_array(dim="band").plot.imshow(
    vmin=0, vmax=0.2
)
plt.title(f"Sentinel-2 on {common_date}")
plt.show()

##############################################################################
# Plot venus
# -------------------------------------------
landsat.sel(time=common_date, method="nearest")[["red", "green", "blue"]].to_array(
    dim="band"
).plot.imshow(vmin=0, vmax=0.2)
plt.title(f"Landsat on {common_date}")

plt.show()
#

##############################################################################
# Plot the fusion
# -------------------------------------------

supercube.sel(time=common_date)[["red", "green", "blue"]].to_array(
    dim="band"
).plot.imshow(vmin=0, vmax=0.2)
plt.title(f"Fusion of Venus/Sentinel-2 on {common_date}")

plt.show()
