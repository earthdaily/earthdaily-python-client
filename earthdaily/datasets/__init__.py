import os
import geopandas as gpd

__pathFile = os.path.dirname(os.path.realpath(__file__))


def _load_json(path, to_wkt: bool = False, to_geojson: bool = False):
    pivot = gpd.read_file(path)
    if to_wkt:
        pivot = pivot.to_wkt()["geometry"].iloc[0]
    if to_geojson:
        pivot = pivot.to_json()
    return pivot


def load_pivot(to_wkt: bool = False, to_geojson: bool = False):
    """
    A pivot located in Nebraska.

    Parameters
    ----------
    to_wkt : BOOL, optional
        Returns the pivot as a wkt. The default is False.
    to_geojson : BOOL, optional
        Returns the pivot as a geojson. The default is False.

    Returns
    -------
    pivot : str, GeoDataFrame
        DESCRIPTION.

    """
    return _load_json(
        os.path.join(__pathFile, f"data{os.path.sep}pivot.geojson"),
        to_wkt=to_wkt,
        to_geojson=to_geojson,
    )


def load_pivot_corumba(to_wkt: bool = False, to_geojson: bool = False):
    """
    A pivot located in Corumba (between Goianas and Brasilia, Brazil).

    Parameters
    ----------
    to_wkt : BOOL, optional
        Returns the pivot as a wkt. The default is False.
    to_geojson : BOOL, optional
        Returns the pivot as a geojson. The default is False.

    Returns
    -------
    pivot : str, GeoDataFrame
        DESCRIPTION.

    """
    return _load_json(
        os.path.join(__pathFile, f"data{os.path.sep}pivot_corumba.geojson"),
        to_wkt=to_wkt,
        to_geojson=to_geojson,
    )
