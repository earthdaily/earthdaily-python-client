from typing import Any

import spyndex
import xarray as xr


def add_indices(dataset: xr.Dataset, indices: list[str], **kwargs: Any) -> xr.Dataset:
    if not kwargs:
        raise ValueError(
            "You must provide band parameters for spectral index calculation. "
            "Example: datacube.add_indices(['NDVI'], R=datacube.data['red'], N=datacube.data['nir'])\n"
            "See spyndex documentation for required parameters: https://github.com/awesome-spectral-indices/spyndex"
        )

    if not indices:
        raise ValueError("At least one index must be provided")

    params = kwargs.copy()
    result_dataset = dataset.copy()

    try:
        idx = spyndex.computeIndex(index=indices, params=params)
    except Exception as e:
        error_msg = str(e)
        available_bands = list(dataset.data_vars)
        provided_params = list(params.keys())

        raise ValueError(
            f"Failed to compute indices {indices}. Error: {error_msg}\n"
            f"Available bands in datacube: {available_bands}\n"
            f"Provided parameters: {provided_params}\n"
            f"See spyndex documentation for required parameters per index"
        ) from e

    if len(indices) == 1:
        idx = idx.expand_dims(index=indices)
    idx = idx.to_dataset(dim="index")

    result_dataset = xr.merge((result_dataset, idx))

    return result_dataset
