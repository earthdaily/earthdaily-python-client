# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:17:30 2023

@author: nkk
"""

import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import Polygon

# Define time, x, and y values
times = ['2022-04-22', '2022-05-12']
x_values = np.arange(0, 8)
y_values = np.arange(0, 3)

# Create 3D arrays for the data values
data_values = np.arange(0, 24).reshape(3, 8)
data_values = np.dstack((data_values,np.full((3,8),4)))

# Create the xarray dataset
ds = xr.Dataset(
    {
        'first_var': (('y', 'x', 'time'), data_values)
    },
    coords={
        'y': y_values,
        'x': x_values,
        'time': times,
    }
).rio.write_crs('EPSG:4326')

# first pixel
geometry = [Polygon([(0,0), (0,0.8), (0.8,0.8), (0.8,0)]),Polygon([(1,1), (9,1), (9,2.1), (1,1)])]
# out of bound geom #            Polygon([(10,10), (10,11), (11,11), (11,10)])]
gdf = gpd.GeoDataFrame({'geometry': geometry}, crs="EPSG:4326")

import earthdaily

zonalstats = earthdaily.earthdatastore.cube_utils.zonal_stats_numpy(ds, gdf, all_touched=True, operations=dict(mean=np.nanmean,max=np.nanmax,min=np.nanmin))
zonalstats['first_var'].sel(stats='min')
zonalstats['first_var'].sel(stats='max')


zonalstats = earthdaily.earthdatastore.cube_utils.zonal_stats(ds,gdf,all_touched=True,operations=['min','max','mean'])
zonalstats['first_var'].sel(stats='min')
zonalstats['first_var'].sel(stats='max')

