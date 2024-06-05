import shapely
import json
import pandas as pd
import geopandas as gpd


class GeometryManager:
    def __init__(self, geometry):
        self.geometry = geometry
        self._obj = self.to_geopandas()

    def __call__(self):
        return self._obj

    def to_intersects(self, crs="EPSG:4326"):
        return json.loads(self._obj.to_crs(crs).dissolve().to_json(drop_id=True))[
            "features"
        ][0]["geometry"]

    def to_wkt(self, crs="EPSG:4326"):
        wkts = list(self._obj.to_crs(crs=crs).to_wkt()["geometry"])
        return wkts[0] if len(wkts) == 1 else wkts

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
            self.input_type = "GeoJson"
            try:
                return gpd.read_file(geometry, driver="GeoJson", crs="EPSG:4326")
            except:
                try:
                    return gpd.read_file(
                        json.dumps(geometry), driver="GeoJson", crs="EPSG:4326"
                    )
                except:
                    pass

            try:
                return gpd.GeoDataFrame.from_features(geometry, crs="EPSG:4326")
            except:
                if "type" in geometry:
                    geom = shapely.__dict__[geometry["type"]](
                        [geometry["coordinates"][0]]
                    )
                    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

        elif isinstance(geometry, gpd.GeoSeries):
            self.input_type = "GeoSeries"

            return gpd.GeoDataFrame(
                geometry=geometry,
                crs="EPSG:4326" if geometry.crs is None else geometry.crs,
            )
        else:
            raise NotImplementedError("Couldn't guess your geometry type")

    def buffer_in_meter(self, distance: int, crs_meters: str = "EPSG:3857", **kwargs):
        return (
            self._obj.to_crs(crs=crs_meters)
            .buffer(distance=distance, **kwargs)
            .to_crs(crs=self._obj.crs)
            .geometry
        )
