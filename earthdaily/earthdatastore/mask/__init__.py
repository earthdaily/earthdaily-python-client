import xarray as xr
import dask
from rasterio.features import geometry_mask
from earthdaily.earthdatastore.cube_utils import _bbox_to_intersects
import geopandas as gpd
import warnings
import numpy as np
import tqdm
from joblib import Parallel, delayed
import psutil

dask.config.set(**{"array.slicing.split_large_chunks": True})

_available_masks = [
    "native",
    "venus_detailed_cloud_mask",
    "ag_cloud_mask",
    "cloud_mask",
    "ag-cloud-mask",
    "cloud_mask_ag_version",
    "cloudmask_ag_version",
    "cloudmask",
    "scl",
]
_native_mask_def_mapping = {
    "sentinel-2-l2a": "scl",
    "venus-l2a": "venus_detailed_cloud_mask",
    "landsat-c2l2-sr": "landsat_qa_pixel",
    "landsat-c2l2-st": "landsat_qa_pixel",
}
_native_mask_asset_mapping = {
    "sentinel-2-l2a": "scl",
    "venus-l2a": "detailed_cloud_mask",
    "landsat-c2l2-sr": "qa_pixel",
    "landsat-c2l2-st": "qa_pixel",
}


class Mask:
    def __init__(self, dataset: xr.Dataset, intersects=None, bbox=None):
        self._obj = dataset
        if bbox and intersects is None:
            intersects = _bbox_to_intersects(bbox)
        if isinstance(intersects, gpd.GeoDataFrame):
            intersects = intersects.to_crs(self._obj.rio.crs)
        self.intersects = intersects
        self.compute_available_pixels()

    def cloud_mask(
        self,
        mask_statistics=False,
    ):
        self._obj["time"] = self._obj.time.dt.round("s")  # rm nano second
        #
        self._obj = self._obj.where(self._obj["cloud_mask"] == 1)
        if mask_statistics:
            self.compute_clear_coverage(
                self._obj["ag_cloud_mask"],
                "ag_cloud_mask",
                1,
                labels_are_clouds=False,
            )
        return self._obj

    def ag_cloud_mask(
        self,
        mask_statistics=False,
    ):
        self._obj["time"] = self._obj.time.dt.round("s")  # rm nano second
        #
        self._obj = self._obj.where(self._obj["ag_cloud_mask"] == 1)
        if mask_statistics:
            self.compute_clear_coverage(
                self._obj["ag_cloud_mask"], "ag_cloud_mask", 1, labels_are_clouds=False
            )
        return self._obj

    def cloudmask_from_asset(
        self,
        cloud_asset,
        labels,
        labels_are_clouds,
        mask_statistics=False,
        fill_value=np.nan,
    ):
        if cloud_asset not in self._obj.data_vars:
            raise ValueError(f"Asset '{cloud_asset}' needed to compute cloudmask.")
        else:
            cloud_layer = self._obj[cloud_asset].copy()
        _assets = [a for a in self._obj.data_vars if a != cloud_asset]

        if fill_value:
            if labels_are_clouds:
                self._obj = self._obj.where(
                    ~self._obj[cloud_asset].isin(labels), fill_value
                )
            else:
                self._obj = self._obj.where(
                    self._obj[cloud_asset].isin(labels), fill_value
                )
        if mask_statistics:
            self.compute_clear_coverage(
                cloud_layer, cloud_asset, labels, labels_are_clouds=labels_are_clouds
            )
        return self._obj

    def scl(
        self,
        clouds_labels=[1, 3, 8, 9, 10, 11],
        mask_statistics=False,
    ):
        return self.cloudmask_from_asset(
            cloud_asset="scl",
            labels=clouds_labels,
            labels_are_clouds=True,
            mask_statistics=mask_statistics,
        )

    def venus_detailed_cloud_mask(self, mask_statistics=False):
        return self.cloudmask_from_asset(
            "detailed_cloud_mask",
            0,
            labels_are_clouds=False,
            mask_statistics=mask_statistics,
        )

    def compute_clear_coverage(
        self, cloudmask_array, cloudmask_name, labels, labels_are_clouds=True
    ):
        if self._obj.attrs.get("usable_pixels", None) is None:
            self.compute_available_pixels()

        n_pixels_as_labels = cloudmask_array.isin(labels).sum(dim=("x", "y")).values
        if labels_are_clouds:
            n_pixels_as_labels = self._obj.attrs["usable_pixels"] - n_pixels_as_labels

        self._obj = self._obj.assign_coords(
            {"clear_pixels": ("time", n_pixels_as_labels)}
        )

        self._obj = self._obj.assign_coords(
            {
                "clear_percent": (
                    "time",
                    np.multiply(
                        n_pixels_as_labels / self._obj.attrs["usable_pixels"],
                        100,
                    ).astype(np.int8),
                )
            }
        )

        return self._obj

    def compute_available_pixels(self):
        if self.intersects is None:
            raise ValueError(
                "bbox or intersects must be defined for now to compute cloud statistics."
            )

        clip_mask_arr = geometry_mask(
            geometries=self.intersects.geometry,
            out_shape=(int(self._obj.rio.height), int(self._obj.rio.width)),
            transform=self._obj.rio.transform(recalc=True),
            all_touched=False,
        )
        self.clip_mask_arr = clip_mask_arr
        usable_pixels = np.sum(np.in1d(clip_mask_arr, False))
        self._obj.attrs["usable_pixels"] = usable_pixels
        return self._obj

    def landsat_qa_pixel(self, mask_statistics=False):
        self._landsat_qa_pixel_convert()
        return self.cloudmask_from_asset(
            "qa_pixel",
            1,
            labels_are_clouds=False,
            mask_statistics=mask_statistics,
        )

    def _landsat_qa_pixel_convert(self):
        # load all time series to fasten next step
        if psutil.virtual_memory().available > self._obj["qa_pixel"].nbytes:
            self._obj["qa_pixel"] = self._obj["qa_pixel"].load()
        for time in self._obj.time:
            data = self._obj["qa_pixel"].loc[dict(time=time)].load().data
            data_f = data.flatten()
            clm = QA_PIXEL_cloud_detection(data_f[~np.isnan(data_f)])
            clm = np.where(clm == 0, np.nan, clm)
            data_f[~np.isnan(data_f)] = clm
            data = data_f.reshape(*data.shape)
            self._obj["qa_pixel"].loc[dict(time=time)] = data


def _QA_PIXEL_cloud_detection(pixel):
    """
    return 1 if cloudfree, 0 is cloud pixel
    """

    px_bin = np.binary_repr(pixel)
    if len(px_bin) == 15:
        reversed_bin = "0" + px_bin[::-1]
    elif len(px_bin) < 15:
        reversed_bin = "0000000000000000"
    else:
        reversed_bin = px_bin[::-1]

    if reversed_bin[7] == "0":
        return 0
    else:
        return 1


def QA_PIXEL_cloud_detection(arr):
    """
    return 1 if cloudfree, 0 is cloud pixel
    """
    uniques = np.unique(arr).astype(np.int16)
    has_cloud = np.array([_QA_PIXEL_cloud_detection(i) for i in uniques])
    cloudfree = np.where(has_cloud == 1, uniques, 0)
    cloudfree_pixels = cloudfree[cloudfree != 0]
    cloudmask = np.isin(arr, cloudfree_pixels).astype(int)
    return cloudmask


def filter_clear_cover(dataset, clear_cover, coordinate="clear_percent"):
    return dataset.isel(time=dataset[coordinate] >= clear_cover)
