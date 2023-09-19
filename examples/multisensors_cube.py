"""
Create a multisensor cube
=================================================================

With Sentinel-2 and Venus, using Sentinel-2 spatial resolutino for demo purpose"""


##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily import earthdatastore
import geopandas as gpd
from rasterio.enums import Resampling
from matplotlib import pyplot as plt


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


s2 = eds.datacube(
    "sentinel-2-l2a",
    intersects=polygon,
    datetime=datetime,
    assets=["blue", "green", "red", "nir"],
    prefer_http=True,
)

##############################################################################
# Generate venus cube
# -------------------------------------------

venus_assets = dict(
    image_file_SRE_B3="blue",
    image_file_SRE_B4="green",
    image_file_SRE_B5="yellow",
    image_file_SRE_B7="red",
    image_file_SRE_B11="nir",
)

venus = eds.datacube(
    "venus-l2a",
    intersects=polygon,
    resolution=s2.rio.resolution()[0],
    datetime=datetime,
    epsg=s2.rio.crs.to_epsg(),
    resampling=Resampling.nearest,  # cubic
    assets=venus_assets,
    prefer_http=True,
)


##############################################################################
# Create supercube
# -------------------------------------------

print("create metacube")
supercube = earthdatastore.metacube(s2, venus)

##############################################################################
# Get the first common date for plotting
# -------------------------------------------

common_date = [
    day
    for day in s2.time.dt.strftime("%Y%m%d").values
    if day in venus.time.dt.strftime("%Y%m%d").values
][0]

##############################################################################
# Plot sentinel-2
# -------------------------------------------

s2.sel(time=common_date)[["red", "green", "blue"]].to_array(dim="band").plot.imshow(
    vmin=0, vmax=0.15
)
plt.title(f"Sentinel-2 on {common_date}")
plt.show()

##############################################################################
# Plot venus
# -------------------------------------------
venus.sel(time=common_date, method="nearest")[["red", "green", "blue"]].to_array(
    dim="band"
).plot.imshow(vmin=0, vmax=0.15)
plt.title(f"Venus on {common_date}")

plt.show()
#

##############################################################################
# Plot the fusion
# -------------------------------------------

supercube.sel(time=common_date)[["red", "green", "blue"]].to_array(
    dim="band"
).plot.imshow(vmin=0, vmax=0.15)
plt.title(f"Fusion of Venus/Sentinel-2 on {common_date}")

plt.show()
