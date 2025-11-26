import unittest

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from earthdaily.datacube._geometry import bbox_to_geopandas, geometry_to_geopandas
from earthdaily.datacube.constants import DEFAULT_BBOX_CRS


class TestGeometryToGeopandas(unittest.TestCase):
    def test_geodataframe_input(self):
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        result = geometry_to_geopandas(gdf)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertIs(result, gdf)

    def test_geoseries_input(self):
        gs = gpd.GeoSeries([box(0, 0, 1, 1)], crs="EPSG:4326")
        result = geometry_to_geopandas(gs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    def test_geoseries_without_crs(self):
        gs = gpd.GeoSeries([box(0, 0, 1, 1)])
        result = geometry_to_geopandas(gs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(result.crs, DEFAULT_BBOX_CRS)

    def test_geoseries_with_crs(self):
        gs = gpd.GeoSeries([box(0, 0, 1, 1)], crs="EPSG:3857")
        result = geometry_to_geopandas(gs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(result.crs, "EPSG:3857")

    def test_featurecollection_dict(self):
        feature_collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                    "properties": {},
                }
            ],
        }
        result = geometry_to_geopandas(feature_collection)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.crs, DEFAULT_BBOX_CRS)

    def test_feature_dict(self):
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        }
        result = geometry_to_geopandas(feature)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.crs, DEFAULT_BBOX_CRS)

    def test_polygon_dict(self):
        polygon_dict = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
        result = geometry_to_geopandas(polygon_dict)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.crs, DEFAULT_BBOX_CRS)

    def test_multipolygon_dict(self):
        multipolygon_dict = {
            "type": "MultiPolygon",
            "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]],
        }
        result = geometry_to_geopandas(multipolygon_dict)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    def test_point_dict(self):
        point_dict = {"type": "Point", "coordinates": [0, 0]}
        result = geometry_to_geopandas(point_dict)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    def test_multipoint_dict(self):
        multipoint_dict = {"type": "MultiPoint", "coordinates": [[0, 0], [1, 1]]}
        result = geometry_to_geopandas(multipoint_dict)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    def test_linestring_dict(self):
        linestring_dict = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
        result = geometry_to_geopandas(linestring_dict)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    def test_multilinestring_dict(self):
        multilinestring_dict = {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1]]]}
        result = geometry_to_geopandas(multilinestring_dict)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    def test_json_string(self):
        polygon_json = '{"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}'
        result = geometry_to_geopandas(polygon_json)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    def test_wkt_string(self):
        wkt_string = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
        result = geometry_to_geopandas(wkt_string)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.crs, DEFAULT_BBOX_CRS)

    def test_invalid_string(self):
        invalid_string = "not a valid geometry"
        with self.assertRaises(ValueError):
            geometry_to_geopandas(invalid_string)

    def test_unsupported_type(self):
        with self.assertRaises(ValueError) as context:
            geometry_to_geopandas(123)
        self.assertIn("Unsupported geometry type", str(context.exception))

    def test_empty_dict(self):
        empty_dict = {}
        with self.assertRaises(ValueError):
            geometry_to_geopandas(empty_dict)

    def test_dict_without_type(self):
        invalid_dict = {"coordinates": [[0, 0], [1, 1]]}
        with self.assertRaises(ValueError):
            geometry_to_geopandas(invalid_dict)


class TestBboxToGeopandas(unittest.TestCase):
    def test_bbox_list(self):
        bbox = [0.0, 0.0, 1.0, 1.0]
        crs = "EPSG:4326"
        result = bbox_to_geopandas(bbox, crs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.crs, crs)

    def test_bbox_tuple(self):
        bbox = (0.0, 0.0, 1.0, 1.0)
        crs = "EPSG:4326"
        result = bbox_to_geopandas(bbox, crs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.crs, crs)

    def test_bbox_string(self):
        bbox = "0.0,0.0,1.0,1.0"
        crs = "EPSG:4326"
        result = bbox_to_geopandas(bbox, crs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.crs, crs)

    def test_bbox_string_with_spaces(self):
        bbox = "0.0, 0.0, 1.0, 1.0"
        crs = "EPSG:4326"
        result = bbox_to_geopandas(bbox, crs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)

    def test_bbox_negative_coordinates(self):
        bbox = [-10.0, -20.0, 10.0, 20.0]
        crs = "EPSG:4326"
        result = bbox_to_geopandas(bbox, crs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        bounds = result.total_bounds
        self.assertEqual(bounds[0], -10.0)
        self.assertEqual(bounds[1], -20.0)
        self.assertEqual(bounds[2], 10.0)
        self.assertEqual(bounds[3], 20.0)

    def test_bbox_three_elements_raises_error(self):
        bbox = [0.0, 0.0, 1.0]
        crs = "EPSG:4326"
        with self.assertRaises(ValueError) as context:
            bbox_to_geopandas(bbox, crs)
        self.assertIn("bbox must have 4 elements", str(context.exception))

    def test_bbox_five_elements_raises_error(self):
        bbox = [0.0, 0.0, 1.0, 1.0, 2.0]
        crs = "EPSG:4326"
        with self.assertRaises(ValueError) as context:
            bbox_to_geopandas(bbox, crs)
        self.assertIn("bbox must have 4 elements", str(context.exception))

    def test_bbox_empty_list_raises_error(self):
        bbox = []
        crs = "EPSG:4326"
        with self.assertRaises(ValueError) as context:
            bbox_to_geopandas(bbox, crs)
        self.assertIn("bbox must have 4 elements", str(context.exception))

    def test_bbox_creates_box_geometry(self):
        bbox = [0.0, 0.0, 1.0, 1.0]
        crs = "EPSG:4326"
        result = bbox_to_geopandas(bbox, crs)
        geom = result.geometry.iloc[0]
        self.assertTrue(geom.equals(box(0.0, 0.0, 1.0, 1.0)))

    def test_bbox_different_crs(self):
        bbox = [0.0, 0.0, 1.0, 1.0]
        crs = "EPSG:3857"
        result = bbox_to_geopandas(bbox, crs)
        self.assertEqual(result.crs, crs)

    def test_bbox_string_float_conversion(self):
        bbox = "0.5,1.5,2.5,3.5"
        crs = "EPSG:4326"
        result = bbox_to_geopandas(bbox, crs)
        bounds = result.total_bounds
        np.testing.assert_array_almost_equal(bounds, [0.5, 1.5, 2.5, 3.5])


if __name__ == "__main__":
    unittest.main()
