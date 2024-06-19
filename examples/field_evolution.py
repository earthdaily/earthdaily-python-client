"""
Field evolution and zonal stats
=================================================================

Using Agriculture cloud mask from EarthDaily, and data from L2A, zonal stats for evolution"""

##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily import earthdatastore, datasets
import geopandas as gpd
from matplotlib import pyplot as plt

##############################################################################
# Load plot
# -------------------------------------------

# load geojson
pivot = datasets.load_pivot()

##############################################################################
# Init earthdatastore with env params
# -------------------------------------------

eds = earthdatastore.Auth()

wkt = "POLYGON ((-114.995866 52.045233, -114.993462 52.046183, -114.988397 52.04159, -114.991187 52.040745, -114.995866 52.045233))"

##############################################################################
# Search for collection items in august 2022 (1st to 9th)
# where at least 50% of the field is clear according to the native cloudmask.

pivot_cube = eds.datacube(
    "sentinel-2-l2a",
    intersects=wkt,
    datetime="2020-06",
    assets=["red", "green", "blue", "swir16"],
    mask_with=["cloud_mask"],
    clear_cover=50,
    search_kwargs=dict(query={"eo:cloud_cover":{"lt":90}}),
    chunks=dict(time=-1)
)
pivot_cube.clear_percent.plot.scatter(x="time")

##############################################################################
# Plots cube with SCL with at least 50% of clear data
# ----------------------------------------------------
pivot_cube = pivot_cube.load()

# Setting the background color of the plot 
# using set_facecolor() method
pivot_cube = pivot_cube.ed.add_indices(['NDSI'])
test = pivot_cube.where(~(pivot_cube['NDSI']>=0))
# test.ed.plot_rgb(col_wrap=3, white_background=False, vmin=0,vmax=0.2)
pivot_cube.ed.plot_rgb(col_wrap=10, vmin=0,vmax=0.15, background=0)
# test.ed.plot_rgb(col_wrap=10, white_background=False, vmin=0,vmax=0.15)

 
# plt.title("Pivot evolution masked with CNN cloudmask filtered with NDSI")
# plt.show()


# ##############################################################################
# # Compute zonal stats for the pivot
# # ----------------------------------------------------

# zonal_stats = pivot_cube.ed.zonal_stats(pivot, ['mean','max','min'])

# zonal_stats.isel(feature=0).to_array(dim="band").plot.line(
#     x="time", col="band", hue="stats"
# )
