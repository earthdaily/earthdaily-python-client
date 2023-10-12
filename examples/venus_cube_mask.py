"""
Venus datacube with native mask
=================================================================

According to a theia location and max cloud cover, using earthdatastore."""

##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily import earthdatastore


##############################################################################
# Init earthdatastore with env params
# -------------------------------------------

eds = earthdatastore.Auth()

##############################################################################
# Set parameters
# -------------------------------------------
collection = "venus-l2a"
theia_location = "MEAD"
max_cloud_cover = 20

query = {
    "theia:location": {"eq": theia_location},
    "eo:cloud_cover": {"lt": max_cloud_cover},
}

##############################################################################
# .. note::
#   Instead of giving list of assets, you can provide a dict, with
#   key as the asset you want, and the value as the name you want.

venus = eds.datacube(
    collection,
    search_kwargs=dict(query=query),
    assets={
        "image_file_SRE_B3": "blue",
        "image_file_SRE_B4": "green",
        "image_file_SRE_B7": "red",
    },
    epsg="32614",
    datetime="2018-11",
    resolution=5,
    mask_with="native",
)

venus.isel(x=slice(4000, 4500), y=slice(4000, 4500))[["red", "green", "blue"]].to_array(
    dim="band"
).plot.imshow(vmin=0, vmax=0.33, col="time")

##############################################################################
# Create the datacube in one function
# -------------------------------------------
