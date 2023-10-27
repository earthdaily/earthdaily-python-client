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

from earthdaily import earthdatastore

##############################################################################
# Import librairies
# -------------------------------------------

eds = earthdatastore.Auth()
polygon = gpd.read_file("pivot.geojson")
# 500x500m
polygon.geometry = (
    polygon.geometry.to_crs(epsg=3857).centroid.buffer(500).to_crs(epsg=4326)
)

datetime = ["2019-08"]


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

venus = eds.datacube(
    "venus-l2a",
    intersects=polygon,
    resolution=s2.rio.resolution()[0],
    datetime=datetime,
    epsg=s2.rio.crs.to_epsg(),
    resampling=Resampling.nearest,  # cubic
    assets=assets,
)

##############################################################################
# Generate Landsat cube
# -------------------------------------------


landsat = eds.datacube(
    "landsat-c2l2-sr",
    intersects=polygon,
    datetime=datetime,
    resampling=Resampling.nearest,
    epsg=s2.rio.crs.to_epsg(),
    resolution=s2.rio.resolution()[0],
    assets=assets,
)
##############################################################################
# Create supercube
# -------------------------------------------

print("create metacube")
supercube = earthdatastore.metacube(s2, venus, landsat)

##############################################################################
# Get the first common date between S2 and Venus for plotting
# ---------------------------------------------------------------

common_date = [
    day
    for day in s2.time.dt.strftime("%Y%m%d").values
    if day in venus.time.dt.strftime("%Y%m%d").values
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
venus.sel(time=common_date, method="nearest")[["red", "green", "blue"]].to_array(
    dim="band"
).plot.imshow(vmin=0, vmax=0.2)
plt.title(f"Venus on {common_date}")

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
