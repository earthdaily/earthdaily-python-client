import warnings
import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from dask import array as da
import spyndex
from dask_image import ndfilters as ndimage

from xarray.core.extensions import AccessorRegistrationWarning

warnings.filterwarnings("ignore", category=AccessorRegistrationWarning)


class MisType(Warning):
    pass


_SUPPORTED_DTYPE = [int, float, list, bool, str]


def _typer(raise_mistype=False):
    def decorator(func):
        def force(*args, **kwargs):
            _args = list(args)
            idx = 1
            for key, val in func.__annotations__.items():
                is_kwargs = key in kwargs.keys()
                if val not in _SUPPORTED_DTYPE or kwargs.get(key, None) is None and is_kwargs or len(args)==1:
                    continue
                if raise_mistype and (val != type(kwargs.get(key)) if is_kwargs else val != type(args[idx])):
                    if is_kwargs:
                        expected = f"{type(kwargs[key]).__name__} ({kwargs[key]})"
                    else:
                        expected = f"{type(args[idx]).__name__} ({args[idx]})"

                        raise MisType(
                        f"{key} expected a {val.__name__}, not a {expected}."
                    )
                if is_kwargs:
                    kwargs[key] = val(kwargs[key]) if val != list else [kwargs[key]]
                else:
                    _args[idx] = val(args[idx]) if val != list else [args[idx]]
                idx+=1
            args = tuple(_args)
            return func(*args, **kwargs)

        return force

    return decorator


@_typer()
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


@_typer()
def _lee_filter(img, window_size: int):
    try:
        from dask_image import ndfilters
    except ImportError:
        raise ImportError("Please install dask-image to run lee_filter")

    img_ = img.copy()
    ndimage_type = ndfilters
    if hasattr(img, "data"):
        if isinstance(img.data, (memoryview, np.ndarray)):
            ndimage_type = ndimage
        img = img.data
    # print(ndimage_type)
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

@xr.register_dataarray_accessor("ed")
class EarthDailyAccessorDataArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def _max_time_wrap(self, wish=5):
        return np.min((wish,self._obj['time'].size))

    @_typer()
    def plot_band(self, cmap="Greys", col="time", col_wrap=5, **kwargs):
        return self._obj.plot.imshow(cmap=cmap, col=col, col_wrap=self._max_time_wrap(col_wrap), **kwargs)

    @_typer()
    def plot_index(
        self, cmap="RdYlGn", vmin=-1, vmax=1, col="time", col_wrap=5, **kwargs
    ):
        return self._obj.plot.imshow(
            vmin=vmin, vmax=vmax, cmap=cmap, col=col, col_wrap=self._max_time_wrap(col_wrap), **kwargs
        )


@xr.register_dataset_accessor("ed")
class EarthDailyAccessorDataset:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _max_time_wrap(self, wish=5):
        return np.min((wish,self._obj['time'].size))
        
    
    @_typer()
    def plot_rgb(
        self,
        red: str = "red",
        green: str = "green",
        blue: str = "blue",
        col="time",
        col_wrap=5,
        **kwargs,
    ):
        return (
            self._obj[[red, green, blue]]
            .to_array(dim="bands")
            .plot.imshow(col=col, col_wrap=self._max_time_wrap(col_wrap), **kwargs)
        )

    @_typer()
    def plot_band(self, band, cmap="Greys", col="time", col_wrap=5, **kwargs):
        return self._obj[band].plot.imshow(
            cmap=cmap, col=col, col_wrap=self._max_time_wrap(col_wrap), **kwargs
        )

    @_typer()
    def plot_index(
        self, index, cmap="RdYlGn", vmin=-1, vmax=1, col="time", col_wrap=5, **kwargs
    ):
        return self._obj[index].plot.imshow(
            vmin=vmin, vmax=vmax, cmap=cmap, col=col, col_wrap=self._max_time_wrap(col_wrap), **kwargs
        )

    @_typer()
    def lee_filter(self, window_size: int = 7):
        return xr.apply_ufunc(
            _lee_filter,
            self._obj,
            input_core_dims=[["time"]],
            dask="allowed",
            output_core_dims=[["time"]],
            kwargs=dict(window_size=window_size),
        )

    @_typer()
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

    def available_index(self, details=False):
        mapper = list(self._auto_mapper().keys())
        indices = spyndex.indices
        available_indices = []
        for k, v in indices.items():
            needed_bands = v.bands
            for needed_band in needed_bands:
                if needed_band not in mapper:
                    break
                available_indices.append(spyndex.indices[k] if details else k)
        return available_indices

    @_typer()
    def add_index(self, index: list, **kwargs):
        """
        Uses spyndex to compute and add index.

        For list of indices, see https://github.com/awesome-spectral-indices/awesome-spectral-indices.


        Parameters
        ----------
        index : list
            ['NDVI'].
        Returns
        -------
        xr.Dataset
            The input xr.Dataset with new data_vars of indices.

        """

        params = {}
        params = self._auto_mapper()
        params.update(**kwargs)
        idx = spyndex.computeIndex(index=index, params=params, **kwargs)

        if len(index) == 1:
            idx = idx.expand_dims(index=index)
        idx = idx.to_dataset(dim="index")

        return xr.merge((self._obj, idx))

    @_typer()
    def sel_nearest_dates(
        self,
        target,
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
        if return_target:
            method_convert = {"bfill": "ffill", "ffill": "bfill", "nearest": "nearest"}
            return self._obj.sel(time=pos), target.sel(
                time=pos, method=method_convert[method]
            )
        return self._obj.sel(time=pos)
