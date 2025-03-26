"""
Compare Sentinel-2 scale/offset evolution during time
=================================================================

Using SCL data from L2A"""

##############################################################################
# Import librairies
# -------------------------------------------

from matplotlib import pyplot as plt

from earthdaily import EarthDataStore, datasets

##############################################################################
# Load plot
# -------------------------------------------

# load geojson
pivot = datasets.load_pivot()

##############################################################################
# Init earthdatastore with environment variables or default credentials
# ----------------------------------------------------------------------------

eds = EarthDataStore()


##############################################################################
# Search for collection items
#
def get_cube(rescale=True):
    pivot_cube = eds.datacube(
        "sentinel-2-l2a",
        intersects=pivot,
        datetime=["2022-01-01", "2022-03-10"],
        assets=["red", "green", "blue"],
        mask_with="native",  # same as scl
        clear_cover=50,  # at least 50% of the polygon must be clear
        rescale=rescale,
    )
    return pivot_cube


##############################################################################
# Get cube with rescale (*0.0001)
# ----------------------------------------------------

pivot_cube = get_cube(rescale=False) * 0.0001

##############################################################################
# Plots cube with SCL with at least 50% of clear data
# ----------------------------------------------------

pivot_cube.ed.plot_rgb(col_wrap=3)
plt.show()

##############################################################################
# Get cube with automatic rescale (default option)
# ----------------------------------------------------

pivot_cube = get_cube()
pivot_cube.clear_percent.plot.scatter(x="time")
plt.show()

##############################################################################
# Plots cube with SCL with at least 50% of clear data
# ----------------------------------------------------

pivot_cube.ed.plot_rgb(col_wrap=3)

plt.show()
