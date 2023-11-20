import shapely
import json
import pandas as pd
import geopandas as gpd


class GeometryManager:
    def __init__(self, geometry):
        self.geometry = geometry
        self._obj = self.to_geopandas()

    def to_intersects(self, crs="EPSG:4326"):
        return json.loads(self._obj.to_crs(crs).dissolve().to_json(drop_id=True))[
            "features"
        ][0]["geometry"]

    def to_wkt(self, crs="EPSG:4326"):
        list(self._obj.to_crs(crs=crs).to_wkt()["geometry"])

    def to_json(self, crs="EPSG:4326"):
        return json.loads(self._obj.to_crs(crs=crs).to_json(drop_id=True))

    def to_geopandas(self):
        return self._unknow_geometry_to_geodataframe(self.geometry)

    def to_bbox(self, crs="EPSG:4326"):
        return self._obj.to_crs(crs=crs).total_bounds

    def _unknow_geometry_to_geodataframe(self, geometry):
        # if isinstance(geometry, pd.DataFrame):
        #     if geometry.size==1:
        #         return gpd.GeoDataFrame(gpd.GeoSeries.from_wkt(geometry.iloc[0]))
        if isinstance(geometry, gpd.GeoDataFrame):
            self.input_type = "GeoDataFrame"
            return geometry
        if isinstance(geometry, str):
            try:
                self.input_type = "wkt"
                return gpd.GeoDataFrame(
                    geometry=[shapely.wkt.loads(geometry)], crs="EPSG:4326"
                )
            except:
                pass
        if isinstance(geometry, (dict, str)):
            if isinstance(geometry, str):
                geometry = json.loads(geometry)
            self.input_type = "geojson"
            try:
                return gpd.GeoDataFrame.from_features(geometry, crs="EPSG:4326")
            except:
                if "type" in geometry:
                    geom = shapely.__dict__[geometry["type"]](
                        geometry["coordinates"][0]
                    )
                    return gpd.GeoDataFrame(geometry=[geom])
        elif isinstance(geometry, gpd.GeoSeries):
            self.input_type = "GeoSeries"
            return gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
        else:
            raise NotImplementedError("Couldn't guess your geometry type")
