"""
Base classes for EarthDaily collection helpers.

This module provides simple base functionality for collection-specific helpers
that make it easier to work with satellite data collections.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pystac
import shapely

from earthdaily import EDSClient
from earthdaily._eds_logging import LoggerConfig
from earthdaily.datacube import Datacube

logger = LoggerConfig(logger_name=__name__).get_logger()


@dataclass
class SpatioTemporalGeometry:
    """Spatiotemporal geometry representation."""

    crs: str
    geometry: shapely.Geometry
    time_range: tuple[datetime, datetime]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "crs": self.crs,
            "geometry": shapely.to_geojson(self.geometry),
            "time_range": [self.time_range[0].isoformat(), self.time_range[1].isoformat()],
        }


class CollectionHelper:
    """Base class for EarthDaily platform collection helpers."""

    def __init__(self, client: EDSClient, collection_id: str, asset_mapping: dict[str, list[str]]) -> None:
        """Initialize collection helper.

        Args:
            client: EarthDaily client instance
            collection_id: STAC collection identifier
            asset_mapping: Mapping from asset keys to band names
        """
        self.client = client
        self.collection_id = collection_id
        self.asset_mapping = asset_mapping

    def get_items(
        self,
        geometries: Optional[list[SpatioTemporalGeometry]] = None,
        intersects: Optional[Union[str, Dict[str, Any]]] = None,
        bbox: Optional[Union[List[float], tuple]] = None,
        datetime: Optional[Union[str, list[str]]] = None,
        max_items: int = 100,
        **search_kwargs,
    ) -> list[pystac.Item]:
        """Get items using the client's search functionality.

        Args:
            geometries: list of SpatioTemporalGeometry objects
            intersects: Geometry to intersect with
            bbox: Bounding box to search within
            datetime: Datetime range for filtering
            max_items: Maximum number of items to return
            **search_kwargs: Additional search parameters

        Returns:
            list of pystac.Item
        """
        if geometries:
            if len(geometries) > 1:
                raise ValueError(
                    f"Only one geometry is supported per search. Got {len(geometries)} geometries. "
                    "Call get_items separately for each geometry."
                )
            if intersects is not None:
                raise ValueError(
                    "Cannot specify both 'geometries' and 'intersects'. "
                    "Use 'geometries' for SpatioTemporalGeometry objects or 'intersects' for raw GeoJSON."
                )
            geometry = geometries[0]
            intersects = geometry.geometry.__geo_interface__
            if not datetime and geometry.time_range:
                start_time = geometry.time_range[0].isoformat()
                end_time = geometry.time_range[1].isoformat()
                datetime = f"{start_time}/{end_time}"
        search_params = {"collections": [self.collection_id], "max_items": max_items, **search_kwargs}

        if intersects:
            search_params["intersects"] = intersects
        if bbox:
            search_params["bbox"] = bbox
        if datetime:
            search_params["datetime"] = datetime
        search = self.client.platform.pystac_client.search(**search_params)
        return list(search.items())

    def download_assets(
        self, items, asset_keys: Optional[list[str]] = None, output_dir: Union[str, Path] = ".", **kwargs
    ) -> dict[str, str]:
        """Download assets using the client's download functionality."""
        downloaded_files = {}

        for item in items:
            try:
                item_data = item.to_dict() if hasattr(item, "to_dict") else item
                result = self.client.platform.stac_item.download_assets(
                    item=item_data,
                    asset_keys=asset_keys,
                    output_dir=str(output_dir),
                    **kwargs,
                )
                downloaded_files.update(result)
            except Exception as e:
                item_id = getattr(item, "id", item.get("id") if isinstance(item, dict) else "unknown")
                logger.warning(f"Error downloading assets for {item_id}: {e}")
                continue

        return downloaded_files

    def create_datacube(
        self,
        items: Optional[list[pystac.Item]] = None,
        geometries: Optional[list[SpatioTemporalGeometry]] = None,
        intersects: Optional[Union[str, Dict[str, Any]]] = None,
        bbox: Optional[Union[List[float], tuple]] = None,
        datetime: Optional[Union[str, list[str]]] = None,
        assets: Optional[list[str]] = None,
        max_items: int = 100,
        **kwargs,
    ) -> Datacube:
        """
        Create datacube using the datacube service.

        Args:
            items: List of STAC items to use directly
            geometries: List of SpatioTemporalGeometry objects
            intersects: Geometry to intersect with
            bbox: Bounding box to search within
            datetime: Datetime range for filtering
            assets: Assets to include in datacube
            max_items: Maximum number of items to search for
            **kwargs: Additional parameters for datacube creation
                - datacube_kwargs: Additional parameters for datacube creation
                - search_kwargs: Additional parameters for search

        Returns:
            Datacube: A Datacube instance wrapping the xarray Dataset
        """
        datacube_kwargs = kwargs.get("datacube_kwargs", {})

        if assets is None:
            assets = list(self.asset_mapping.keys())

        if items is None:
            items = self.get_items(
                geometries=geometries,
                intersects=intersects,
                bbox=bbox,
                datetime=datetime,
                max_items=max_items,
                **kwargs.get("search_kwargs", {}),
            )

        return self.client.datacube.create(
            items=items,
            assets=assets,
            bbox=bbox,
            intersects=intersects,
            **datacube_kwargs,
        )

    def get_available_assets(self) -> list[str]:
        """Get list of available assets for this collection."""
        return list(self.asset_mapping.keys())

    def get_asset_description(self, asset_key: str) -> str:
        """Get human-readable description of an asset."""
        return f"Asset '{asset_key}' with bands: {', '.join(self.asset_mapping.get(asset_key, []))}"
