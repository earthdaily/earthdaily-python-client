import xarray as xr
from xarray.core import groupby
import numpy as np


class CustomOperations:
    @staticmethod
    def _np_mode(arr, **kwargs):
        if isinstance(arr, list):
            arr = np.asarray(arr)
        if isinstance(arr, xr.Dataset | xr.DataArray):
            if arr.chunks is not None:
                arr = arr.compute()
                # or it will output
            # NotImplementedError: Slicing an array with unknown chunks with a dask.array of ints is not supported
        values, counts = np.unique(arr, return_counts=True)
        rm = np.isnan(values)
        values, counts = values[~rm], counts[~rm]
        return values[np.argmax(counts)]

    @staticmethod
    def mode(data_array_grouped, optional_arg=None):
        # Apply _xrmode to DataArrayGroupBy object
        result = data_array_grouped.reduce(
            CustomOperations._np_mode,
            list(dim for dim in data_array_grouped.dims if dim != "time"),
        )
        return result

    @staticmethod
    def register_custom_operations():
        # register custom methods fo DataArrayGroupBy
        xr.core.groupby.DataArrayGroupBy.mode = CustomOperations.mode
        xr.core.groupby.DatasetGroupBy.mode = CustomOperations.mode
        np.mode = CustomOperations._np_mode


CustomOperations.register_custom_operations()
