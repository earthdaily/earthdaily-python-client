"""
Datacube creation with odc-stac or stackstac
=================================================================

Just choose your engine !"""

##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily import earthdatastore
import geopandas as gpd
from matplotlib import pyplot as plt

##############################################################################
# Init earthdaily
# -------------------------------------------

eds = earthdatastore.Auth()
bbox = "1.2235878938028861,43.642464388086324,1.254365582002663,43.65799608988672"

##############################################################################
# One way (item+datacube at the same time)
# -------------------------------------------

datacube = eds.datacube(
    "sentinel-2-l2a",
    bbox=bbox,
    datetime="2023-04",
    assets=["red", "green", "blue"],
    mask_with="scl",
)
datacube.isel(time=0).to_array(dim="band").plot.imshow(vmin=0, vmax=0.23)
plt.show()


##############################################################################
# stackstac

datacube = eds.datacube(
    "sentinel-2-l2a",
    bbox=bbox,
    datetime="2023-04",
    assets=["red", "green", "blue"],
    engine="stackstac",  # or "odc", default one
    mask_with="scl",
)
datacube.isel(time=0).to_array(dim="band").plot.imshow(vmin=0, vmax=0.23)
plt.show()
