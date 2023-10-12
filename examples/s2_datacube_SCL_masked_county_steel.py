"""
Generate a Sentinel-2 Datacube with SCL cloudmask
===============================================================

Using EarthDataStore."""

##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily import earthdatastore
import geopandas as gpd
from matplotlib import pyplot as plt

##########################
# Loading geometry

geometry = gpd.read_file("county_steel.geojson")
geometry.geometry = geometry.centroid.to_crs(epsg=3857).buffer(1000)

##########################
# Init earthdaily and check available assets

eds = earthdatastore.Auth()  # using config from ENV
available_assets = eds.explore("sentinel-2-l2a").assets()
print(available_assets)

###########################################################
# Create datacube from earthdatastore
# ----------------------------------------
s2_datacube = eds.datacube(
    "sentinel-2-l2a",
    assets=["blue", "green", "red", "nir"],
    intersects=geometry,
    datetime=["2022-07"],
    mask_with="native",  # equal to "scl" for sentinel-2
    mask_statistics=True,
)

s2_datacube.clear_percent_scl.plot.scatter(x="time")
plt.title("Percentage of clear pixels on the study site")
plt.show()
print(s2_datacube)

s2_datacube = s2_datacube.load()  # load in memory

s2_datacube[["red", "green", "blue"]].to_array(dim="band").plot.imshow(
    vmin=0, vmax=0.22, col="time", col_wrap=4
)

###########################################################
# Compute zonal stats
# ----------------------------------------

geometry_stats = earthdatastore.cube_utils.zonal_stats(
    s2_datacube, geometry, operations=["mean"]
)

print(geometry_stats)

###########################################################
# Plot mean values
# ----------------------------------------
# Plot mean values from the first feature.

geometry_stats.isel(feature=0).sel(stats="mean").to_array(dim="band").plot.line(
    x="time", col="band"
)
