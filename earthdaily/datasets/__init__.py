import os
import geopandas as gpd

__pathFile = os.path.dirname(os.path.realpath(__file__))


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
    pivot = gpd.read_file(os.path.join(__pathFile, f"data{os.path.sep}pivot.geojson"))
    if to_wkt:
        pivot = pivot.to_wkt()["geometry"].iloc[0]
    if to_geojson:
        pivot = pivot.to_json()
    return pivot
