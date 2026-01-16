"""
Sentinel-2 Collection Helper Example
==============================

This example demonstrates how to use the new Sentinel2CollectionHelper class to access
and work with Sentinel-2 L2A (Level-2A) data through the EarthDaily platform.

Features demonstrated:
- Creating Sentinel-2 collection helpers with different band combinations
- Searching for items with cloud cover filtering
- Analyzing spectral and processing information
- Downloading assets
- Creating datacubes and calculating vegetation indices

Requirements:
- Set your EDS credentials as environment variables or in a .env file
- Install with collection helpers support: pip install "earthdaily[legacy,platform]"
"""

from datetime import datetime, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Consider installing python-dotenv to automatically load .env files:")
    print("   pip install python-dotenv")

import numpy as np
import shapely

from earthdaily import EDSClient, EDSConfig
from earthdaily.exceptions import EDSAPIError
from earthdaily.platform.helpers.collections import Sentinel2CollectionHelper
from earthdaily.platform.helpers.collections.base import SpatioTemporalGeometry


def initialize_client():
    """Initialize the EarthDaily API client."""
    print("Initializing EarthDaily Client...")
    config = EDSConfig()
    client = EDSClient(config)
    print("Client initialized successfully!")
    return client


def create_test_geometry():
    """Create a test geometry for demonstration."""
    print("Creating test geometry...")

    center_lon, center_lat = -93.5, 41.8
    size = 0.1

    geometry = shapely.box(center_lon - size / 2, center_lat - size / 2, center_lon + size / 2, center_lat + size / 2)

    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)

    st_geometry = SpatioTemporalGeometry(crs="EPSG:4326", geometry=geometry, time_range=(start_time, end_time))

    print(f"   Area: {geometry.bounds}")
    print(f"   Time range: {start_time.date()} to {end_time.date()}")

    return st_geometry


def demonstrate_collection_helper_creation(client):
    """Demonstrate different ways to create Sentinel-2 collection helpers."""
    print("Demonstrating Sentinel-2 Collection Helper Creation")
    print("-" * 50)

    # 1. Default collection helper (all bands)
    print("\n1. Creating default Sentinel-2 collection helper:")
    s2_default = Sentinel2CollectionHelper(client)
    print(f"   {s2_default}")
    print(f"   Total assets: {len(s2_default.get_available_assets())}")

    # 2. RGB collection helper
    print("\n2. Creating RGB collection helper:")
    s2_rgb = Sentinel2CollectionHelper.create_rgb(client)
    print(f"   {s2_rgb}")
    print(f"   Assets: {s2_rgb.assets}")

    # 3. Vegetation analysis collection helper
    print("\n3. Creating vegetation analysis collection helper:")
    s2_veg = Sentinel2CollectionHelper.create_vegetation(client)
    print(f"   {s2_veg}")
    print(f"   Assets: {s2_veg.assets}")

    # 4. Agriculture collection helper
    print("\n4. Creating agriculture collection helper:")
    s2_ag = Sentinel2CollectionHelper.create_agriculture(client)
    print(f"   {s2_ag}")
    print(f"   Assets: {s2_ag.assets}")

    # 5. Analysis-ready collection helper
    print("\n5. Creating analysis-ready collection helper:")
    s2_analysis = Sentinel2CollectionHelper.create_analysis_ready(client)
    print(f"   {s2_analysis}")
    print(f"   Assets: {s2_analysis.assets}")

    # 6. Custom with cloud cover threshold
    print("\n6. Creating custom collection helper with cloud cover threshold:")
    s2_custom = Sentinel2CollectionHelper(
        client, assets=["red", "green", "blue", "nir", "swir16"], cloud_cover_threshold=20.0
    )
    print(f"   {s2_custom}")
    print(f"   Cloud cover threshold: {s2_custom.cloud_cover_threshold}%")

    return s2_analysis  # Return analysis-ready for further examples


def search_sentinel2_items(collection_helper, geometry):
    """Search for Sentinel-2 items."""
    print("Searching for Sentinel-2 Items")
    print("-" * 40)

    try:
        print("\n1. Basic search:")
        items = collection_helper.get_items(geometries=[geometry], max_items=15)
        print(f"   Found {len(items)} items")

        if items:
            for i, item in enumerate(items[:3]):
                print(f"   {i + 1}. {item.id}")
                print(f"      Time: {item.datetime.date() if item.datetime else 'N/A'}")

        print("\n2. Search with low cloud cover (<10%):")
        clear_items = collection_helper.get_items(geometries=[geometry], cloud_cover_max=10.0, max_items=10)

        clear_count = len(clear_items)
        print(f"   Found {clear_count} items with <10% cloud cover")

        print("\n3. Search for recent items (last 30 days):")
        recent_end = datetime.now()
        recent_start = recent_end - timedelta(days=30)
        datetime_range = f"{recent_start.isoformat()}/{recent_end.isoformat()}"

        recent_items = collection_helper.get_items(geometries=[geometry], datetime=datetime_range, max_items=10)

        recent_count = len(recent_items)
        print(f"   Found {recent_count} recent items")

        return items[:8]

    except Exception as e:
        print(f"Error searching for items: {e}")
        return []


def analyze_item_metadata(collection_helper, items):
    """Analyze metadata of found items."""
    if not items:
        print("No items to analyze")
        return

    print("Analyzing Item Metadata")
    print("-" * 30)

    print("\n1. Cloud Cover Analysis:")
    cloud_info = collection_helper.get_cloud_cover_info(items)

    cloud_values = [cc for cc in cloud_info.values() if cc is not None]
    if cloud_values:
        print(f"   Average cloud cover: {np.mean(cloud_values):.1f}%")
        print(f"   Min/Max cloud cover: {np.min(cloud_values):.1f}% / {np.max(cloud_values):.1f}%")

        for item_name, cloud_cover in list(cloud_info.items())[:3]:
            print(f"   {item_name}: {cloud_cover}%")

    print("\n2. Processing Information:")
    proc_info = collection_helper.get_processing_info(items)

    for item_name, proc_data in list(proc_info.items())[:3]:
        mgrs_tile = proc_data.get("mgrs_tile", "Unknown")
        baseline = proc_data.get("processing_baseline", "Unknown")
        print(f"   {item_name}:")
        print(f"     MGRS Tile: {mgrs_tile}")
        print(f"     Processing Baseline: {baseline}")

    print("\n3. Filtering by Cloud Cover:")
    clear_items = collection_helper.filter_by_cloud_cover(items, 15.0)
    very_clear_items = collection_helper.filter_by_cloud_cover(items, 5.0)

    print(f"   Items with <15% cloud cover: {len(clear_items)}")
    print(f"   Items with <5% cloud cover: {len(very_clear_items)}")

    return clear_items if clear_items else items


def demonstrate_band_information():
    """Demonstrate band information capabilities."""
    print("Demonstrating Band Information")
    print("-" * 40)

    key_bands = ["blue", "green", "red", "nir", "rededge1", "swir16"]

    for band in key_bands:
        band_info = Sentinel2CollectionHelper.get_band_info(band)
        print(f"\n{band.upper()}:")
        print(f"   Band ID: {band_info.get('band_id', 'N/A')}")
        print(f"   Wavelength: {band_info.get('center_wavelength', 'N/A')}nm")
        print(f"   Resolution: {band_info.get('resolution', 'N/A')}m")
        print(f"   Description: {band_info.get('description', 'N/A')}")

    print("\nAvailable band combinations:")
    combinations = Sentinel2CollectionHelper.BAND_COMBINATIONS
    for name, bands in list(combinations.items())[:5]:
        print(f"   {name}: {bands}")


def demonstrate_asset_downloads(collection_helper, items, output_dir="./downloads"):
    """Demonstrate asset downloading."""
    if not items:
        print("No items available for download")
        return

    print("Demonstrating Asset Downloads")
    print("-" * 35)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        print("\n1. Downloading thumbnails:")
        thumbnail_downloads = collection_helper.download_assets(
            items=items[:1],
            asset_keys=["thumbnail"],
            output_dir=output_path / "sentinel2_thumbnails",
        )

        print(f"   Downloaded {len(thumbnail_downloads)} thumbnail files")
        for asset_key, file_path in thumbnail_downloads.items():
            print(f"     {asset_key}: {file_path}")

        print("\n2. Downloading visual (TCI) images:")
        visual_downloads = collection_helper.download_assets(
            items=items[:1],
            asset_keys=["visual"],
            output_dir=output_path / "sentinel2_visual",
        )

        print(f"   Downloaded {len(visual_downloads)} visual files")

        download_bands = input("\n   Download spectral bands (RGB+NIR)? (y/N): ").lower() == "y"

        if download_bands:
            print("\n3. Downloading RGB+NIR bands:")
            band_downloads = collection_helper.download_assets(
                items=items[:1],
                asset_keys=["red"],
                output_dir=output_path / "sentinel2_bands",
            )

            print(f"   Downloaded {len(band_downloads)} band files")
            for asset_key, file_path in band_downloads.items():
                print(f"     {asset_key}: {file_path}")

    except Exception as e:
        print(f"Error downloading assets: {e}")


def create_sentinel2_datacube(client, geometry):
    """Create a Sentinel-2 datacube and calculate vegetation indices."""
    print("Creating Sentinel-2 Datacube")
    print("-" * 35)

    try:
        s2_source = Sentinel2CollectionHelper.create_vegetation(client, cloud_cover_threshold=20.0)

        items = s2_source.get_items(geometries=[geometry], cloud_cover_max=15.0, max_items=8)

        if not items:
            print("   No clear items found for datacube creation")
            return None
        print(f"   Using {len(items)} items for datacube")

        datacube = s2_source.create_datacube(
            items=items, assets=["red", "green", "blue", "nir", "rededge1"], apply_cloud_mask=True
        )

        print("Datacube created successfully!")
        print(f"   Shape: {datacube.shape}")
        print(f"   Bands: {datacube.bands}")
        if datacube.timestamps:
            print(f"   Time range: {datacube.timestamps[0]} to {datacube.timestamps[-1]}")

        if "red" in datacube.bands and "nir" in datacube.bands:
            print("Calculating vegetation indices:")

            datacube_with_ndvi = s2_source.calculate_ndvi(datacube)
            ndvi = datacube_with_ndvi.data["NDVI"]
            print("   NDVI statistics:")
            print(f"     Mean: {float(ndvi.mean()):.3f}")
            print(f"     Std: {float(ndvi.std()):.3f}")
            print(f"     Min/Max: {float(ndvi.min()):.3f} / {float(ndvi.max()):.3f}")

        if "green" in datacube.bands and "nir" in datacube.bands:
            datacube_with_ndwi = s2_source.calculate_ndwi(datacube)
            ndwi = datacube_with_ndwi.data["NDWI"]
            print("   NDWI statistics:")
            print(f"     Mean: {float(ndwi.mean()):.3f}")
            print(f"     Std: {float(ndwi.std()):.3f}")
            print(f"     Min/Max: {float(ndwi.min()):.3f} / {float(ndwi.max()):.3f}")

        print("Band statistics:")
        for var in ["red", "green", "blue", "nir"]:
            if var in datacube.bands:
                band_data = datacube.data[var]
                print(f"   {var.upper()}: mean={float(band_data.mean()):.4f}, std={float(band_data.std()):.4f}")

        return datacube

    except Exception as e:
        print(f"Error creating datacube: {e}")
        return None


def main():
    try:
        print("Sentinel-2 Collection Helper Example")
        print("=" * 40)

        client = initialize_client()
        geometry = create_test_geometry()
        s2_collection_helper = demonstrate_collection_helper_creation(client)
        demonstrate_band_information()
        items = search_sentinel2_items(s2_collection_helper, geometry)
        filtered_items = analyze_item_metadata(s2_collection_helper, items)

        if filtered_items:
            download_demo = input("\n   Run download demonstration? (y/N): ").lower() == "y"
            if download_demo:
                demonstrate_asset_downloads(s2_collection_helper, filtered_items)

        datacube = create_sentinel2_datacube(client, geometry)
        print(f"Datacube: {datacube}")
        print("Sentinel-2 example completed successfully!")

    except EDSAPIError as e:
        print(f"API Error: {e}")
        print(f"   Status Code: {e.status_code}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
