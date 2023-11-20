"""
Venus datacube
=================================================================

According to a theia location and max cloud cover, using earthdatastore."""

##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily import earthdatastore

##############################################################################
# Load credentials and init earthdatastore
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
# Search for items
# -------------------------------------------


##############################################################################
# Search for items
# -------------------------------------------

items = eds.search(collection, query=query, prefer_alternate="download")

##############################################################################
# .. note::
#   We specify prefer_http=True because we didn't set any s3 credentials.


print(f"{theia_location} venus location has {len(items)} items.")

##############################################################################
# Create the datacube and bandname mapping
# -------------------------------------------


##############################################################################
# .. note::
#   As transform and other metadata are missing in assets,
#   compute them for first asset

epsg, resolution = (
    items[0].properties["proj:epsg"],
    items[0].properties["gsd"],
)

##############################################################################
# .. note::
#   Instead of giving list of assets, you can provide a dict, with
#   key as the asset you want, and the value as the name you want.


venus_datacube = earthdatastore.datacube(
    items, assets=["blue", "green", "red"], epsg=epsg, resolution=resolution
)
print(venus_datacube)

venus_datacube.isel(time=slice(29, 31), x=slice(4000, 4500), y=slice(4000, 4500))[
    ["red", "green", "blue"]
].to_array(dim="band").plot.imshow(col="time", vmin=0, vmax=0.30)
