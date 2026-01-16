"""
Sentinel-1 collection helper for EarthDaily platform.

This module provides a helper class for working with Sentinel-1 RTC data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pystac

from earthdaily import EDSClient
from earthdaily.datacube import Datacube
from earthdaily.platform.helpers.collections.base import CollectionHelper, SpatioTemporalGeometry


class Sentinel1CollectionHelper:
    """
    Collection helper for Sentinel-1 RTC data on EarthDaily platform.

    This class provides easy access to Sentinel-1 Radiometrically Terrain Corrected (RTC) data.
    """

    COLLECTION_ID = "sentinel-1-rtc"

    ASSET_BANDS = {
        "vv": ["VV"],
        "vh": ["VH"],
        "hh": ["HH"],
        "rendered_preview": ["rendered_preview"],
    }

    BAND_COMBINATIONS = {
        "dual_pol": ["vv", "vh"],
        "single_pol_vv": ["vv"],
        "single_pol_vh": ["vh"],
        "single_pol_hh": ["hh"],
    }

    def __init__(self, client: EDSClient, assets: Optional[list[str]] = None, **kwargs) -> None:
        """
        Initialize Sentinel-1 collection helper.

        Args:
            client: EarthDaily client instance
            assets: list of asset keys to work with (None for all available)
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

    @classmethod
    def create_dual_pol(cls, client: EDSClient, **kwargs) -> Sentinel1CollectionHelper:
        """Create Sentinel-1 helper configured for dual polarization (VV + VH)."""
        return cls(client, assets=cls.BAND_COMBINATIONS["dual_pol"], **kwargs)

    @classmethod
    def create_single_pol_vv(cls, client: EDSClient, **kwargs) -> Sentinel1CollectionHelper:
        """Create Sentinel-1 helper configured for VV polarization only."""
        return cls(client, assets=cls.BAND_COMBINATIONS["single_pol_vv"], **kwargs)

    @classmethod
    def create_single_pol_vh(cls, client: EDSClient, **kwargs) -> Sentinel1CollectionHelper:
        """Create Sentinel-1 helper configured for VH polarization only."""
        return cls(client, assets=cls.BAND_COMBINATIONS["single_pol_vh"], **kwargs)

    @classmethod
    def create_single_pol_hh(cls, client: EDSClient, **kwargs) -> Sentinel1CollectionHelper:
        """Create Sentinel-1 helper configured for HH polarization only."""
        return cls(client, assets=cls.BAND_COMBINATIONS["single_pol_hh"], **kwargs)

    def get_items(
        self,
        geometries: Optional[list[SpatioTemporalGeometry]] = None,
        intersects: Optional[Union[str, Dict[str, Any]]] = None,
        bbox: Optional[Union[List[float], tuple]] = None,
        datetime: Optional[Union[str, list[str]]] = None,
        instrument_mode: Optional[str] = None,
        platform: Optional[str] = None,
        max_items: int = 100,
        **search_kwargs,
    ) -> list[pystac.Item]:
        """
        Get Sentinel-1 items with optional filtering.

        Args:
            geometries: list of SpatioTemporalGeometry objects
            intersects: Geometry to intersect with
            bbox: Bounding box to search within
            datetime: Datetime range for filtering
            instrument_mode: Filter by SAR instrument mode (e.g., 'IW', 'EW', 'SM')
            platform: Filter by platform (e.g., 'sentinel-1a', 'sentinel-1b')
            max_items: Maximum number of items to return
            **search_kwargs: Additional search parameters

        Returns:
            list of STAC items
        """
        if "query" not in search_kwargs:
            search_kwargs["query"] = {}

        if instrument_mode:
            search_kwargs["query"]["sar:instrument_mode"] = {"eq": instrument_mode}
        if platform:
            search_kwargs["query"]["platform"] = {"eq": platform}

        if not search_kwargs["query"]:
            del search_kwargs["query"]

        return self._core.get_items(
            geometries=geometries,
            intersects=intersects,
            bbox=bbox,
            datetime=datetime,
            max_items=max_items,
            **search_kwargs,
        )

    def create_datacube(
        self,
        items: Optional[list[pystac.Item]] = None,
        geometries: Optional[list[SpatioTemporalGeometry]] = None,
        intersects: Optional[Union[str, Dict[str, Any]]] = None,
        bbox: Optional[Union[List[float], tuple]] = None,
        datetime: Optional[Union[str, list[str]]] = None,
        assets: Optional[list[str]] = None,
        instrument_mode: Optional[str] = None,
        platform: Optional[str] = None,
        max_items: int = 100,
        **datacube_kwargs,
    ) -> Datacube:
        """
        Create datacube with Sentinel-1 specific filtering.

        Args:
            items: List of STAC items to use directly
            geometries: List of SpatioTemporalGeometry objects
            intersects: Geometry to intersect with
            bbox: Bounding box to search within
            datetime: Datetime range for filtering
            assets: Assets to include in datacube
            instrument_mode: Filter by SAR instrument mode (e.g., 'IW', 'EW', 'SM')
            platform: Filter by platform (e.g., 'sentinel-1a', 'sentinel-1b')
            max_items: Maximum number of items to search for
            **datacube_kwargs: Additional parameters for datacube creation

        Returns:
            Datacube: A Datacube instance wrapping the xarray Dataset
        """
        assets = assets or self.assets
        search_kwargs: dict[str, Any] = {}
        query: dict[str, Any] = {}

        if instrument_mode:
            query["sar:instrument_mode"] = {"eq": instrument_mode}
        if platform:
            query["platform"] = {"eq": platform}

        if query:
            search_kwargs["query"] = query

        return self._core.create_datacube(
            items=items,
            geometries=geometries,
            intersects=intersects,
            bbox=bbox,
            datetime=datetime,
            assets=assets,
            max_items=max_items,
            search_kwargs=search_kwargs,
            datacube_kwargs=datacube_kwargs,
        )

    def get_polarization_info(self, items) -> dict[str, list[str]]:
        """Get polarization information for items."""
        polarization_info = {}
        for item in items:
            available_pols = []
            if "vv" in item.assets or any("vv" in asset_key.lower() for asset_key in item.assets.keys()):
                available_pols.append("VV")
            if "vh" in item.assets or any("vh" in asset_key.lower() for asset_key in item.assets.keys()):
                available_pols.append("VH")
            if "hh" in item.assets or any("hh" in asset_key.lower() for asset_key in item.assets.keys()):
                available_pols.append("HH")
            polarization_info[item.id] = available_pols
        return polarization_info

    def get_orbit_info(self, items) -> dict[str, dict[str, Any]]:
        """Get orbit information for items."""
        orbit_info = {}
        for item in items:
            orbit_data = {
                "orbit_direction": item.properties.get("sat:orbit_state"),
                "orbit_number": item.properties.get("sat:absolute_orbit"),
                "relative_orbit": item.properties.get("sat:relative_orbit"),
                "platform": item.properties.get("platform"),
                "instrument": item.properties.get("instruments", []),
            }
            orbit_info[item.id] = orbit_data
        return orbit_info

    def filter_by_polarization(self, items, required_polarizations: list[str]) -> list:
        """
        Filter items by required polarizations.

        Args:
            items: List of STAC items
            required_polarizations: List of required polarizations (e.g., ["VV", "VH"])

        Returns:
            List of items that have all required polarizations
        """
        filtered_items = []
        for item in items:
            available_pols = []
            if "vv" in item.assets or any("vv" in asset_key.lower() for asset_key in item.assets.keys()):
                available_pols.append("VV")
            if "vh" in item.assets or any("vh" in asset_key.lower() for asset_key in item.assets.keys()):
                available_pols.append("VH")
            if "hh" in item.assets or any("hh" in asset_key.lower() for asset_key in item.assets.keys()):
                available_pols.append("HH")

            if all(pol in available_pols for pol in required_polarizations):
                filtered_items.append(item)

        return filtered_items

    @staticmethod
    def get_asset_description(asset_key: str) -> str:
        """Get human-readable description of an asset."""
        descriptions = {
            "vv": "VV polarization: vertical transmit, vertical receive (gamma naught RTC)",
            "vh": "VH polarization: vertical transmit, horizontal receive (gamma naught RTC)",
            "hh": "HH polarization: horizontal transmit, horizontal receive (gamma naught RTC)",
            "rendered_preview": "Rendered preview image (PNG thumbnail)",
        }
        return descriptions.get(asset_key, f"Asset: {asset_key}")

    def __repr__(self) -> str:
        return f"Sentinel1CollectionHelper(collection='{self.COLLECTION_ID}', assets={self.assets})"

    def get_available_assets(self) -> list[str]:
        return self._core.get_available_assets()

    # Assets that use direct href instead of alternate.download.href
    DIRECT_HREF_ASSETS = {"rendered_preview", "tilejson"}

    def download_assets(
        self, items, asset_keys: Optional[list[str]] = None, output_dir: str = ".", **kwargs
    ) -> dict[str, str]:
        """
        Download assets, handling different href types appropriately.

        Some assets (like rendered_preview) use direct href, while data assets
        (vv, vh, hh) use alternate.download.href with signed URLs.
        """
        if asset_keys is None:
            asset_keys = self.assets

        # Separate assets by href type
        direct_href_keys = [k for k in asset_keys if k in self.DIRECT_HREF_ASSETS]
        alternate_href_keys = [k for k in asset_keys if k not in self.DIRECT_HREF_ASSETS]

        downloaded_files: dict[str, str] = {}

        # Download assets with alternate.download.href (default behavior)
        if alternate_href_keys:
            result = self._core.download_assets(
                items=items,
                asset_keys=alternate_href_keys,
                output_dir=output_dir,
                **kwargs,
            )
            downloaded_files.update(result)

        # Download assets with direct href
        if direct_href_keys:
            result = self._core.download_assets(
                items=items,
                asset_keys=direct_href_keys,
                output_dir=output_dir,
                href_type="href",  # Use direct href instead of alternate.download.href
                **kwargs,
            )
            downloaded_files.update(result)

        return downloaded_files
