import json

import geopandas as gpd
import shapely


class GeometryManager:
    """
    A class to manage and convert various types of geometries into different formats.

    Parameters
    ----------
    geometry : various types
        Input geometry, can be GeoDataFrame, GeoSeries, WKT string, or GeoJSON.

    Attributes
    ----------
    geometry : various types
        The input geometry provided by the user.
    _obj : GeoDataFrame
        The geometry converted into a GeoDataFrame.
    input_type : str
        Type of input geometry inferred during processing.

    Methods
    -------
    __call__():
        Returns the stored GeoDataFrame.
    to_intersects(crs='EPSG:4326'):
        Converts geometry to GeoJSON intersects format with a specified CRS.
    to_wkt(crs='EPSG:4326'):
        Converts geometry to WKT format with a specified CRS.
    to_json(crs='EPSG:4326'):
        Converts geometry to GeoJSON format with a specified CRS.
    to_geopandas():
        Returns the GeoDataFrame of the input geometry.
    to_bbox(crs='EPSG:4326'):
        Returns the bounding box of the geometry with a specified CRS.
    buffer_in_meter(distance, crs_meters='EPSG:3857', **kwargs):
        Applies a buffer in meters to the geometry and returns it with the original CRS.
    """

    def __init__(self, geometry):
        self.geometry = geometry
        self._obj = self.to_geopandas()

    def __call__(self):
        """Returns the GeoDataFrame stored in the object."""
        return self._obj

    def to_intersects(self, crs="EPSG:4326"):
        """
        Converts the geometry to GeoJSON intersects format.

        Parameters
        ----------
        crs : str, optional
            The coordinate reference system (CRS) to convert to (default is EPSG:4326).

        Returns
        -------
        dict
            The geometry in GeoJSON intersects format.
        """
        return json.loads(self._obj.to_crs(crs).dissolve().geometry.to_json())[
            "features"
        ][0]["geometry"]

    def to_wkt(self, crs="EPSG:4326"):
        """
        Converts the geometry to WKT format.

        Parameters
        ----------
        crs : str, optional
            The CRS to convert to (default is EPSG:4326).

        Returns
        -------
        str or list of str
            The geometry in WKT format. If there is only one geometry, a single WKT
            string is returned; otherwise, a list of WKT strings.
        """
        wkts = list(self._obj.to_crs(crs=crs).to_wkt()["geometry"])
        return wkts[0] if len(wkts) == 1 else wkts

    def to_json(self, crs="EPSG:4326"):
        """
        Converts the geometry to GeoJSON format.

        Parameters
        ----------
        crs : str, optional
            The CRS to convert to (default is EPSG:4326).

        Returns
        -------
        dict
            The geometry in GeoJSON format.
        """
        return json.loads(self._obj.to_crs(crs=crs).to_json(drop_id=True))

    def to_geopandas(self):
        """
        Converts the input geometry to a GeoDataFrame.

        Returns
        -------
        GeoDataFrame
            The input geometry as a GeoDataFrame.
        """
        return self._unknow_geometry_to_geodataframe(self.geometry)

    def to_bbox(self, crs="EPSG:4326"):
        """
        Returns the bounding box of the geometry.

        Parameters
        ----------
        crs : str, optional
            The CRS to convert to (default is EPSG:4326).

        Returns
        -------
        numpy.ndarray
            The bounding box as an array [minx, miny, maxx, maxy].
        """
        return self._obj.to_crs(crs=crs).total_bounds

    def _unknow_geometry_to_geodataframe(self, geometry):
        """
        Attempts to convert an unknown geometry format into a GeoDataFrame.

        Parameters
        ----------
        geometry : various types
            The input geometry, which can be a GeoDataFrame, GeoSeries, WKT string,
            or GeoJSON.

        Returns
        -------
        GeoDataFrame
            The converted geometry as a GeoDataFrame.

        Raises
        ------
        NotImplementedError
            If the geometry type cannot be inferred or converted.
        """
        if isinstance(geometry, gpd.GeoDataFrame):
            self.input_type = "GeoDataFrame"
            return geometry
        if isinstance(geometry, str):
            try:
                self.input_type = "wkt"
                return gpd.GeoDataFrame(
                    geometry=[shapely.wkt.loads(geometry)], crs="EPSG:4326"
                )
            except Exception:
                pass
        if isinstance(geometry, (dict, str)):
            self.input_type = "GeoJson"
            try:
                return gpd.read_file(geometry, driver="GeoJson", crs="EPSG:4326")
            except Exception:
                try:
                    return gpd.read_file(
                        json.dumps(geometry), driver="GeoJson", crs="EPSG:4326"
                    )
                except Exception:
                    pass

            try:
                return gpd.GeoDataFrame.from_features(geometry, crs="EPSG:4326")
            except Exception:
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
        """
        Applies a buffer in meters to the geometry and returns it with the original CRS.

        Parameters
        ----------
        distance : int
            The buffer distance in meters.
        crs_meters : str, optional
            The CRS to use for calculating the buffer (default is EPSG:3857).
        **kwargs : dict, optional
            Additional keyword arguments to pass to the buffer method.

        Returns
        -------
        GeoSeries
            The buffered geometry in the original CRS.
        """
        return (
            self._obj.to_crs(crs=crs_meters)
            .buffer(distance=distance, **kwargs)
            .to_crs(crs=self._obj.crs)
            .geometry
        )
