import json
from typing import Any

import geopandas as gpd
from shapely import wkt
from shapely.geometry import box, shape

from earthdaily.datacube.constants import DEFAULT_BBOX_CRS


def geometry_to_geopandas(geometry: str | dict[str, Any] | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if isinstance(geometry, gpd.GeoDataFrame):
        return geometry

    if isinstance(geometry, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=geometry)
        if gdf.crs is None and geometry.crs is not None:
            gdf.set_crs(geometry.crs, inplace=True)
        elif gdf.crs is None:
            gdf.set_crs(DEFAULT_BBOX_CRS, inplace=True)
        return gdf

    if isinstance(geometry, dict):
        if geometry.get("type") == "FeatureCollection":
            gdf = gpd.GeoDataFrame.from_features(geometry["features"])
            if gdf.crs is None:
                gdf.set_crs(DEFAULT_BBOX_CRS, inplace=True)
            return gdf
        elif geometry.get("type") in [
            "Feature",
            "Polygon",
            "MultiPolygon",
            "Point",
            "MultiPoint",
            "LineString",
            "MultiLineString",
        ]:
            geom = shape(geometry)
            gdf = gpd.GeoDataFrame(geometry=[geom], crs=DEFAULT_BBOX_CRS)
            return gdf

    if isinstance(geometry, str):
        try:
            geom_dict = json.loads(geometry)
            return geometry_to_geopandas(geom_dict)
        except (json.JSONDecodeError, ValueError):
            try:
                geom = wkt.loads(geometry)
                return gpd.GeoDataFrame(geometry=[geom], crs=DEFAULT_BBOX_CRS)
            except Exception:
                pass

    raise ValueError(f"Unsupported geometry type: {type(geometry)}")


def bbox_to_geopandas(bbox: list[float] | tuple[float, ...] | str, crs: str) -> gpd.GeoDataFrame:
    if isinstance(bbox, str):
        bbox = [float(i) for i in bbox.split(",")]

    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements [minx, miny, maxx, maxy], got {len(bbox)}")

    minx, miny, maxx, maxy = bbox
    geom = box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame(geometry=[geom], crs=crs)
