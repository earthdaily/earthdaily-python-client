"""
Sentinel-2 collection helper for EarthDaily platform.

This module provides a helper class for working with Sentinel-2 L2A data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pystac

from earthdaily import EDSClient
from earthdaily.datacube import Datacube
from earthdaily.platform.helpers.collections.base import CollectionHelper, SpatioTemporalGeometry


class Sentinel2CollectionHelper:
    """
    Collection helper for Sentinel-2 L2A data on EarthDaily platform.

    This class provides easy access to Sentinel-2 Level-2A atmospherically corrected data.
    """

    COLLECTION_ID = "sentinel-2-l2a"

    ASSET_BANDS = {
        # Visible bands (10m)
        "blue": ["B02"],  # Blue (490nm) - 10m
        "green": ["B03"],  # Green (560nm) - 10m
        "red": ["B04"],  # Red (665nm) - 10m
        # Near-infrared
        "nir": ["B08"],  # NIR (842nm) - 10m
        "nir08": ["B8A"],  # NIR narrow (865nm) - 20m
        "nir09": ["B09"],  # NIR (945nm) - 60m
        # Red edge bands (20m)
        "rededge1": ["B05"],  # Red Edge 1 (705nm) - 20m
        "rededge2": ["B06"],  # Red Edge 2 (740nm) - 20m
        "rededge3": ["B07"],  # Red Edge 3 (783nm) - 20m
        # SWIR bands (20m)
        "swir16": ["B11"],  # SWIR 1 (1610nm) - 20m
        "swir22": ["B12"],  # SWIR 2 (2190nm) - 20m
        # Atmospheric bands (60m)
        "coastal": ["B01"],  # Coastal aerosol (443nm) - 60m
        # Derived products
        "visual": ["visual"],  # True color image (TCI) - 10m
        "scl": ["scl"],  # Scene Classification Layer - 20m
        "aot": ["aot"],  # Aerosol Optical Thickness - 20m
        "wvp": ["wvp"],  # Water Vapour - 20m
        # Preview
        "thumbnail": ["thumbnail"],  # JPEG preview image
    }

    BAND_COMBINATIONS = {
        # Standard RGB
        "rgb": ["red", "green", "blue"],
        "true_color": ["red", "green", "blue"],
        # False color composites
        "false_color_nir": ["nir", "red", "green"],
        "false_color_swir": ["swir16", "nir", "red"],
        # Vegetation analysis
        "vegetation": ["nir", "red", "green", "rededge1", "rededge2", "rededge3"],
        "ndvi_bands": ["nir", "red"],
        "red_edge": ["rededge1", "rededge2", "rededge3", "nir"],
        # Water analysis
        "water": ["green", "nir", "swir16", "swir22"],
        # Agriculture
        "agriculture": ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "swir16"],
        # All 10m bands
        "bands_10m": ["blue", "green", "red", "nir"],
        # All 20m bands
        "bands_20m": ["rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"],
        # All spectral bands (excluding derived products)
        "all_spectral": [
            "coastal",
            "blue",
            "green",
            "red",
            "rededge1",
            "rededge2",
            "rededge3",
            "nir",
            "nir08",
            "nir09",
            "swir16",
            "swir22",
        ],
        # Analysis ready (common bands for most applications)
        "analysis_ready": ["blue", "green", "red", "nir", "swir16", "swir22"],
    }

    def __init__(
        self,
        client: EDSClient,
        assets: Optional[list[str]] = None,
        cloud_cover_threshold: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Initialize Sentinel-2 collection helper.

        Args:
            client: EarthDaily client instance
            assets: list of asset keys to work with (None for all available)
            cloud_cover_threshold: Maximum cloud cover percentage (0-100)
            **kwargs: Additional parameters
        """
        if assets is not None:
            asset_mapping = {k: v for k, v in self.ASSET_BANDS.items() if k in assets}
        else:
            asset_mapping = self.ASSET_BANDS.copy()

        self._core = CollectionHelper(
            client=client,
            collection_id=self.COLLECTION_ID,
            asset_mapping=asset_mapping,
        )
        self.assets = assets or list(self.ASSET_BANDS.keys())
        self.cloud_cover_threshold = cloud_cover_threshold

    @classmethod
    def create_rgb(cls, client: EDSClient, **kwargs) -> Sentinel2CollectionHelper:
        """Create Sentinel-2 helper configured for RGB bands."""
        return cls(client, assets=cls.BAND_COMBINATIONS["rgb"], **kwargs)

    @classmethod
    def create_vegetation(cls, client: EDSClient, **kwargs) -> Sentinel2CollectionHelper:
        """Create Sentinel-2 helper configured for vegetation analysis."""
        return cls(client, assets=cls.BAND_COMBINATIONS["vegetation"], **kwargs)

    @classmethod
    def create_agriculture(cls, client: EDSClient, **kwargs) -> Sentinel2CollectionHelper:
        """Create Sentinel-2 helper configured for agricultural applications."""
        return cls(client, assets=cls.BAND_COMBINATIONS["agriculture"], **kwargs)

    @classmethod
    def create_analysis_ready(cls, client: EDSClient, **kwargs) -> Sentinel2CollectionHelper:
        """Create Sentinel-2 helper with analysis-ready bands."""
        return cls(client, assets=cls.BAND_COMBINATIONS["analysis_ready"], **kwargs)

    def get_items(
        self,
        geometries: Optional[list[SpatioTemporalGeometry]] = None,
        intersects: Optional[Union[str, Dict[str, Any]]] = None,
        bbox: Optional[Union[List[float], tuple]] = None,
        datetime: Optional[Union[str, list[str]]] = None,
        cloud_cover_max: Optional[float] = None,
        max_items: int = 100,
        **search_kwargs,
    ) -> list[pystac.Item]:
        """
        Get Sentinel-2 items with cloud cover filtering.

        Args:
            geometries: list of SpatioTemporalGeometry objects
            intersects: Geometry to intersect with
            bbox: Bounding box to search within
            datetime: Datetime range for filtering
            cloud_cover_max: Maximum cloud cover percentage (0-100)
            max_items: Maximum number of items to return
            **search_kwargs: Additional search parameters

        Returns:
            list of STAC items
        """
        max_cloud_cover = cloud_cover_max if cloud_cover_max is not None else self.cloud_cover_threshold
        if max_cloud_cover is not None:
            if "query" not in search_kwargs:
                search_kwargs["query"] = {}
            search_kwargs["query"]["eo:cloud_cover"] = {"lt": max_cloud_cover}

        return self._core.get_items(
            geometries=geometries,
            intersects=intersects,
            bbox=bbox,
            datetime=datetime,
            max_items=max_items,
            **search_kwargs,
        )

    SCL_CLOUD_SHADOW = 3
    SCL_CLOUD_MEDIUM = 8
    SCL_CLOUD_HIGH = 9
    SCL_THIN_CIRRUS = 10

    DEFAULT_EXCLUDE_VALUES = [SCL_CLOUD_SHADOW, SCL_CLOUD_MEDIUM, SCL_CLOUD_HIGH, SCL_THIN_CIRRUS]

    def create_datacube(
        self,
        items: Optional[list[pystac.Item]] = None,
        geometries: Optional[list[SpatioTemporalGeometry]] = None,
        intersects: Optional[Union[str, Dict[str, Any]]] = None,
        bbox: Optional[Union[List[float], tuple]] = None,
        datetime: Optional[Union[str, list[str]]] = None,
        assets: Optional[list[str]] = None,
        cloud_cover_max: Optional[float] = None,
        apply_cloud_mask: bool = True,
        mask_band: str = "scl",
        exclude_values: Optional[list[int]] = None,
        max_items: int = 100,
        **datacube_kwargs,
    ) -> Datacube:
        """
        Create datacube with Sentinel-2 specific filtering and optional cloud masking.

        Args:
            items: List of STAC items to use directly
            geometries: List of SpatioTemporalGeometry objects
            intersects: Geometry to intersect with
            bbox: Bounding box to search within
            datetime: Datetime range for filtering
            assets: Assets to include in datacube
            cloud_cover_max: Maximum cloud cover percentage (0-100)
            apply_cloud_mask: Whether to apply cloud masking using SCL band
            mask_band: Band to use for masking (default: "scl")
            exclude_values: SCL values to exclude (default: cloud shadow, clouds, cirrus)
            max_items: Maximum number of items to search for
            **datacube_kwargs: Additional parameters for datacube creation

        Returns:
            Datacube: A Datacube instance wrapping the xarray Dataset
        """
        assets = assets or self.assets
        max_cloud_cover = cloud_cover_max if cloud_cover_max is not None else self.cloud_cover_threshold
        search_kwargs = {}
        if max_cloud_cover is not None:
            search_kwargs["query"] = {"eo:cloud_cover": {"lt": max_cloud_cover}}

        datacube_assets = list(assets) if assets else []
        if apply_cloud_mask and mask_band not in datacube_assets:
            datacube_assets.append(mask_band)

        datacube = self._core.create_datacube(
            items=items,
            geometries=geometries,
            intersects=intersects,
            bbox=bbox,
            datetime=datetime,
            assets=datacube_assets,
            max_items=max_items,
            search_kwargs=search_kwargs,
            datacube_kwargs=datacube_kwargs,
        )

        if apply_cloud_mask and mask_band in datacube.bands:
            mask_exclude = exclude_values if exclude_values is not None else self.DEFAULT_EXCLUDE_VALUES
            datacube = datacube.apply_mask(
                mask_band=mask_band,
                exclude_values=mask_exclude,
                mask_statistics=True,
            )

        return datacube

    def get_cloud_cover_info(self, items) -> dict[str, float]:
        """Get cloud cover information for items."""
        cloud_cover_info = {}
        for item in items:
            cloud_cover = item.properties.get("eo:cloud_cover")
            cloud_cover_info[item.id] = cloud_cover
        return cloud_cover_info

    def get_processing_info(self, items) -> dict[str, dict[str, Any]]:
        """Get processing information for items."""
        processing_info = {}
        for item in items:
            proc_data = {
                "processing_level": item.properties.get("processing:level"),
                "processing_baseline": item.properties.get("s2:processing_baseline"),
                "product_type": item.properties.get("s2:product_type"),
                "datatake_id": item.properties.get("s2:datatake_id"),
                "granule_id": item.properties.get("s2:granule_id"),
                "mgrs_tile": item.properties.get("s2:mgrs_tile"),
                "platform": item.properties.get("platform"),
                "instruments": item.properties.get("instruments", []),
            }
            processing_info[item.id] = proc_data
        return processing_info

    def filter_by_cloud_cover(self, items, max_cloud_cover: float) -> list:
        """
        Filter items by maximum cloud cover percentage.

        Args:
            items: List of STAC items
            max_cloud_cover: Maximum cloud cover percentage (0-100)

        Returns:
            List of items with cloud cover below the threshold
        """
        filtered_items = []
        for item in items:
            cloud_cover = item.properties.get("eo:cloud_cover")
            if cloud_cover is not None and cloud_cover < max_cloud_cover:
                filtered_items.append(item)
        return filtered_items

    def calculate_ndvi(self, datacube: Datacube) -> Datacube:
        """
        Calculate NDVI from a datacube using the add_indices method.

        Args:
            datacube: Datacube with 'red' and 'nir' bands

        Returns:
            Datacube with NDVI index added
        """
        if "red" not in datacube.bands or "nir" not in datacube.bands:
            raise ValueError("Datacube must contain 'red' and 'nir' bands for NDVI calculation")

        return datacube.add_indices(["NDVI"], R=datacube.data["red"], N=datacube.data["nir"])

    def calculate_ndwi(self, datacube: Datacube) -> Datacube:
        """
        Calculate NDWI (Normalized Difference Water Index) from a datacube.

        Args:
            datacube: Datacube with 'green' and 'nir' bands

        Returns:
            Datacube with NDWI index added
        """
        if "green" not in datacube.bands or "nir" not in datacube.bands:
            raise ValueError("Datacube must contain 'green' and 'nir' bands for NDWI calculation")

        return datacube.add_indices(["NDWI"], G=datacube.data["green"], N=datacube.data["nir"])

    @staticmethod
    def get_band_info(asset_key: str) -> dict[str, Any]:
        """Get detailed information about a spectral band."""
        band_info = {
            # Visible bands
            "blue": {
                "band_id": "B02",
                "center_wavelength": 490,
                "bandwidth": 65,
                "resolution": 10,
                "description": "Blue band - useful for coastal/aerosol studies",
            },
            "green": {
                "band_id": "B03",
                "center_wavelength": 560,
                "bandwidth": 35,
                "resolution": 10,
                "description": "Green band - peak of vegetation reflectance",
            },
            "red": {
                "band_id": "B04",
                "center_wavelength": 665,
                "bandwidth": 30,
                "resolution": 10,
                "description": "Red band - chlorophyll absorption",
            },
            # NIR bands
            "nir": {
                "band_id": "B08",
                "center_wavelength": 842,
                "bandwidth": 115,
                "resolution": 10,
                "description": "NIR band - vegetation structure",
            },
            "nir08": {
                "band_id": "B8A",
                "center_wavelength": 865,
                "bandwidth": 20,
                "resolution": 20,
                "description": "NIR narrow band - vegetation monitoring",
            },
            # Red edge bands
            "rededge1": {
                "band_id": "B05",
                "center_wavelength": 705,
                "bandwidth": 15,
                "resolution": 20,
                "description": "Red Edge 1 - vegetation stress detection",
            },
            "rededge2": {
                "band_id": "B06",
                "center_wavelength": 740,
                "bandwidth": 15,
                "resolution": 20,
                "description": "Red Edge 2 - vegetation health",
            },
            "rededge3": {
                "band_id": "B07",
                "center_wavelength": 783,
                "bandwidth": 20,
                "resolution": 20,
                "description": "Red Edge 3 - vegetation/soil transition",
            },
            # SWIR bands
            "swir16": {
                "band_id": "B11",
                "center_wavelength": 1610,
                "bandwidth": 90,
                "resolution": 20,
                "description": "SWIR 1 - moisture content, vegetation/soil discrimination",
            },
            "swir22": {
                "band_id": "B12",
                "center_wavelength": 2190,
                "bandwidth": 180,
                "resolution": 20,
                "description": "SWIR 2 - mineral discrimination",
            },
        }

        return band_info.get(asset_key, {"description": f"Asset: {asset_key}", "band_id": asset_key})

    def __repr__(self) -> str:
        return f"Sentinel2CollectionHelper(collection='{self.COLLECTION_ID}', assets={len(self.assets)} bands)"

    def get_available_assets(self) -> list[str]:
        return self._core.get_available_assets()

    def download_assets(
        self, items, asset_keys: Optional[list[str]] = None, output_dir: str = ".", **kwargs
    ) -> dict[str, str]:
        return self._core.download_assets(items=items, asset_keys=asset_keys, output_dir=output_dir, **kwargs)
