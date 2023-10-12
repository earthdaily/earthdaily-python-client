"""
Datacube creation with cloudmask and stats
=================================================================

A simple datacube for Sentinel-2 cloudmasked with SCL
"""

##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily import earthdatastore
import geopandas as gpd

##############################################################################
# Init earthdatastore with env params
# -------------------------------------------

eds = earthdatastore.Auth()
bbox = "-123.017178,49.433027,-122.936668,49.471750"

##############################################################################
# One way (item+datacube at the same time)
# -------------------------------------------

datacube = eds.datacube(
    "sentinel-2-l2a",
    datetime="2022-07",
    bbox=bbox,
    assets=["red", "green", "blue"],
    mask_with="native",
    mask_statistics=True,
)

##############################################################################
# Plot clear cover per day
# -------------------------------------------

datacube.clear_percent_scl.plot.scatter(x="time")

##############################################################################
# Plot where clear cover > 50
# -------------------------------------------

datacube.sel(time=datacube.clear_percent_scl > 50).to_array(dim="band").plot.imshow(
    vmin=0, vmax=0.25, col="time", col_wrap=3
)
