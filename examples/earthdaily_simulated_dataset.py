"""
EarthDaily Simulated Dataset
=================================================================

Following a pivot field"""

##############################################################################
# Import librairies
# -------------------------------------------

import earthdaily
import datetime
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt

ProgressBar().register()  # to have chunks progress bar


##############################################################################
# Loading pivot
# -------------------------------------------

pivot = earthdaily.datasets.load_pivot_corumba()

##############################################################################
# Auth to earthdatastore
# -------------------------------------------

eds = earthdaily.earthdatastore.Auth()

##############################################################################
# Define timerange
# -------------------------------------------

delta_days = 10
datetime_list = ["2018-10-01", "2019-04-15"]

##############################################################################
# Request items for vnir dataset
# -------------------------------------------

items = eds.search(
    "earthdaily-simulated-cloudless-l2a-cog-edagro",
    intersects=pivot,
    datetime=datetime_list,
    query={"instruments": {"contains": "vnir"}},
    prefer_alternate="download",
)[::delta_days]  # an keep on item every n delta_days


##############################################################################
# Generate datacube for RGB and NIR
# -------------------------------------------

datacube = earthdaily.earthdatastore.datacube(
    items, intersects=pivot, assets=["blue", "green", "red", "nir"]
).load()

##############################################################################
# Plot RGB image time series
# -------------------------------------------

datacube[["red", "green", "blue"]].to_array(dim="band").plot.imshow(
    col="time", col_wrap=4, vmax=0.2
)

##############################################################################
# Plot mean RGB time series
# -------------------------------------------

datacube[["blue", "green", "red", "nir"]].groupby("time").mean(...).to_array(
    dim="band"
).plot(col="band")

##############################################################################
# Plot NDVI evolution
# -------------------------------------------

datacube["ndvi"] = (datacube["nir"] - datacube["red"]) / (
    datacube["nir"] + datacube["red"]
)

fig, ax = plt.subplots()
mean_ndvi = datacube[["ndvi"]].groupby("time").mean(...).to_array(dim="band")
std_ndvi = datacube[["ndvi"]].groupby("time").std(...).to_array(dim="band")
ax.fill_between(
    mean_ndvi.time,
    (mean_ndvi.values + std_ndvi.values)[0, ...],
    (mean_ndvi.values - std_ndvi.values)[0, ...],
    alpha=0.3,
    color="C1",
)
mean_ndvi.plot(ax=ax, c="C1")
plt.grid(alpha=0.4)
plt.title("NDVI evolution every 10 days")
