import warnings
import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from dask import array as da
import spyndex
from dask_image import ndfilters as dask_ndimage
from scipy import ndimage
from xarray.core.extensions import AccessorRegistrationWarning
from ..earthdatastore.cube_utils import GeometryManager


def xr_loop_func(
    dataset: xr.Dataset,
    func,
    to_numpy: bool = False,
    loop_dimension: str = "time",
    **kwargs,
):
    def _xr_loop_func(dataset, metafunc, loop_dimension, **kwargs):
        if to_numpy is True:
            dataset_func = dataset.copy()
            looped = [
                metafunc(dataset.isel({loop_dimension: i}).load().data, **kwargs)
                for i in range(dataset[loop_dimension].size)
            ]
            dataset_func.data = np.asarray(looped)
            return dataset_func
        else:
            return xr.concat(
                [
                    metafunc(dataset.isel({loop_dimension: i}), **kwargs)
                    for i in range(dataset[loop_dimension].size)
                ],
                dim=loop_dimension,
            )

    return dataset.map(
        func=_xr_loop_func, metafunc=func, loop_dimension=loop_dimension, **kwargs
    )


def _lee_filter(img, window_size: int):
    img_ = img.copy()
    if isinstance(img, np.ndarray):
        ndimage_type = ndimage
    else:
        ndimage_type = dask_ndimage
    binary_nan = ndimage_type.minimum_filter(
        xr.where(np.isnan(img), 0, 1), size=window_size
    )
    binary_nan = np.where(binary_nan == 0, np.nan, 1)
    img = xr.where(np.isnan(img), 0, img)
    window_size = da.from_array([window_size, window_size, 1])

    img_mean = ndimage_type.uniform_filter(img, window_size)
    img_sqr_mean = ndimage_type.uniform_filter(img**2, window_size)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = np.var(img, axis=(0, 1))

    img_weights = img_variance / (np.add(img_variance, overall_variance))

    img_output = img_mean + img_weights * (np.subtract(img, img_mean))
    img_output = xr.where(np.isnan(binary_nan), img_, img_output)
    return img_output


def _xr_rio_clip(datacube, geom):
    geom_ = GeometryManager(geom)
    geom = geom_.to_geopandas().to_crs(datacube.rio.crs)
    datacube = datacube.rio.clip_box(*geom.total_bounds)
    datacube = datacube.rio.clip(geom.geometry)
    return datacube


@xr.register_dataarray_accessor("ed")
class EarthDailyAccessorDataArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def clip(self, geom):
        return _xr_rio_clip(self._obj, geom)

    def _max_time_wrap(self, wish=5, col="time"):
        return np.min((wish, self._obj[col].size))

    def plot_band(self, cmap="Greys", col="time", col_wrap=5, **kwargs):
        return self._obj.plot.imshow(
            cmap=cmap,
            col=col,
            col_wrap=self._max_time_wrap(col_wrap, col=col),
            **kwargs,
        )

    def whittaker(
        self,
        beta: float = 10000.0,
        weights: (np.ndarray, list) = None,
        time="time",
    ):
        from . import whittaker

        return whittaker.whittaker(self._obj, beta=beta, weights=weights, time=time)

    def sel_nearest_dates(
        self,
        target: (xr.Dataset, xr.DataArray),
        max_delta: int = 0,
        method: str = "nearest",
        return_target: bool = False,
    ):
        src_time = self._obj.sel(time=target.time.dt.date, method=method).time.dt.date
        target_time = target.time.dt.date
        pos = np.abs(src_time.data - target_time.data)
        pos = [
            src_time.isel(time=i).time.values
            for i, j in enumerate(pos)
            if j.days <= max_delta
        ]
        pos = np.unique(pos)
        if return_target:
            method_convert = {"bfill": "ffill", "ffill": "bfill", "nearest": "nearest"}
            return self._obj.sel(time=pos), target.sel(
                time=pos, method=method_convert[method]
            )
        return self._obj.sel(time=pos)

    def zonal_stats(
        self,
        geometry,
        operations: list = ["mean"],
        raise_missing_geometry: bool = False,
    ):
        from ..earthdatastore.cube_utils import zonal_stats, GeometryManager

        geometry = GeometryManager(geometry).to_geopandas()
        return zonal_stats(
            self._obj,
            geometry,
            operations=operations,
            raise_missing_geometry=raise_missing_geometry,
        )

    def lee_filter(self, window_size: int):
        return xr.apply_ufunc(
            _lee_filter,
            self._obj,
            input_core_dims=[["time"]],
            dask="allowed",
            output_core_dims=[["time"]],
            kwargs=dict(window_size=window_size),
        )

    def centroid(self, to_wkt: str = False, to_4326: bool = True):
        """Return the geographic center point in 4326/WKT of this dataset."""
        # we can use a cache on our accessor objects, because accessors
        # themselves are cached on instances that access them.
        lon = float(self._obj.x[int(self._obj.x.size / 2)])
        lat = float(self._obj.y[int(self._obj.y.size / 2)])
        point = gpd.GeoSeries([Point(lon, lat)], crs=self._obj.rio.crs)
        if to_4326:
            point = point.to_crs(epsg="4326")
        if to_wkt:
            point = point.map(lambda x: x.wkt).iloc[0]
        return point

    def drop_unfrozen_coords(self, keep_spatial_ref=True):
        unfrozen_coords = [
            coord
            for coord in self._obj.coords
            if coord not in list(self._obj.sizes.keys())
        ]
        if keep_spatial_ref and "spatial_ref" in unfrozen_coords:
            unfrozen_coords.pop(
                np.argwhere(np.in1d(unfrozen_coords, "spatial_ref"))[0][0]
            )
        return self._obj.drop(unfrozen_coords)


@xr.register_dataset_accessor("ed")
class EarthDailyAccessorDataset(EarthDailyAccessorDataArray):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot_rgb(
        self,
        red: str = "red",
        green: str = "green",
        blue: str = "blue",
        col="time",
        col_wrap=5,
        background: None | int | float = None,
        **kwargs,
    ):
        ds = self._obj
        if isinstance(background, int | float):
            ds = xr.where(np.isnan(ds[blue]), background, ds)
        return (
            ds[[red, green, blue]]
            .to_array(dim="bands")
            .plot.imshow(
                col=col, col_wrap=self._max_time_wrap(col_wrap, col=col), **kwargs
            )
        )

    def plot_band(self, band, cmap="Greys", col="time", col_wrap=5, **kwargs):
        return self._obj[band].plot.imshow(
            cmap=cmap,
            col=col,
            col_wrap=self._max_time_wrap(col_wrap, col=col),
            **kwargs,
        )

    def _auto_mapper(self):
        _BAND_MAPPING = {
            "coastal": "A",
            "blue": "B",
            "green": "G",
            "yellow": "Y",
            "red": "R",
            "rededge1": "RE1",
            "rededge2": "RE2",
            "rededge3": "RE3",
            "rededge70": "RE1",
            "rededge74": "RE2",
            "rededge78": "RE3",
            "nir": "N",
            "nir08": "N2",
            "watervapor": "WV",
            "swir16": "S1",
            "swir22": "S2",
            "lwir": "T1",
            "lwir11": "T2",
            "vv": "VV",
            "vh": "VH",
            "hh": "HH",
            "hv": "HV",
        }

        params = {}
        data_vars = list(
            self._obj.rename(
                {var: var.lower() for var in self._obj.data_vars}
            ).data_vars
        )
        for v in data_vars:
            if v in _BAND_MAPPING.keys():
                params[_BAND_MAPPING[v]] = self._obj[v]
        return params

    def available_indices(self, details=False):
        mapper = list(self._auto_mapper().keys())
        indices = spyndex.indices
        available_indices = []
        for k, v in indices.items():
            needed_bands = v.bands
            missing_bands = False
            for needed_band in needed_bands:
                if needed_band not in mapper:
                    missing_bands = True
                    break
            if missing_bands is False:
                available_indices.append(spyndex.indices[k] if details else k)
        return available_indices

    def add_indices(self, indices: list, **kwargs):
        """
        Uses spyndex to compute and add index.

        For list of indices, see https://github.com/awesome-spectral-indices/awesome-spectral-indices.


        Parameters
        ----------
        indices : list
            ['NDVI'].
        Returns
        -------
        xr.Dataset
            The input xr.Dataset with new data_vars of indices.

        """

        params = {}
        params = self._auto_mapper()
        params.update(**kwargs)
        idx = spyndex.computeIndex(index=indices, params=params, **kwargs)

        if len(indices) == 1:
            idx = idx.expand_dims(index=indices)
        idx = idx.to_dataset(dim="index")

        return xr.merge((self._obj, idx))
