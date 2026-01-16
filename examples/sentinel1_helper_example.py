"""
Sentinel-1 Collection Helper Example
==============================

This example demonstrates how to use the new Sentinel1CollectionHelper class to access
and work with Sentinel-1 RTC (Radiometrically Terrain Corrected) data through
the EarthDaily platform.

Features demonstrated:
- Creating Sentinel-1 collection helpers with different configurations
- Searching for items with various filters
- Analyzing polarization and orbit information
- Downloading assets
- Creating datacubes for analysis

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

import shapely

from earthdaily import EDSClient, EDSConfig
from earthdaily.exceptions import EDSAPIError
from earthdaily.platform.helpers.collections import Sentinel1CollectionHelper
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
    size = 0.1  # degrees (roughly 10km)

    geometry = shapely.box(center_lon - size / 2, center_lat - size / 2, center_lon + size / 2, center_lat + size / 2)

    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)

    st_geometry = SpatioTemporalGeometry(crs="EPSG:4326", geometry=geometry, time_range=(start_time, end_time))

    print(f"   Area: {geometry.bounds}")
    print(f"   Time range: {start_time.date()} to {end_time.date()}")

    return st_geometry


def demonstrate_collection_helper_creation(client):
    """Demonstrate different ways to create Sentinel-1 collection helpers."""
    print("Demonstrating Sentinel-1 Collection Helper Creation")
    print("-" * 50)

    print("1. Creating default Sentinel-1 collection helper:")
    s1_default = Sentinel1CollectionHelper(client)
    print(f"   {s1_default}")
    print(f"   Available assets: {s1_default.get_available_assets()}")

    print("2. Creating dual polarization collection helper:")
    s1_dual_pol = Sentinel1CollectionHelper.create_dual_pol(client)
    print(f"   {s1_dual_pol}")
    print(f"   Assets: {s1_dual_pol.assets}")

    print("3. Creating VV-only collection helper:")
    s1_vv = Sentinel1CollectionHelper.create_single_pol_vv(client)
    print(f"   {s1_vv}")
    print(f"   Assets: {s1_vv.assets}")

    print("4. Creating custom collection helper with specific assets:")
    s1_custom = Sentinel1CollectionHelper(client, assets=["vv", "vh"])
    print(f"   {s1_custom}")
    print(f"   Assets: {s1_custom.assets}")

    return s1_dual_pol  # Return dual pol for further examples


def search_sentinel1_items(collection_helper, geometry):
    """Search for Sentinel-1 items."""
    print("Searching for Sentinel-1 Items")
    print("-" * 40)

    try:
        print("1. Basic search:")
        items = collection_helper.get_items(geometries=[geometry], max_items=10)

        print(f"   Found {len(items)} items")

        if items:
            for i, item in enumerate(items[:3]):
                print(f"   {i + 1}. {item.id}")
                print(f"      Time: {item.datetime.date() if item.datetime else 'N/A'}")

        print("2. Search for sentinel-1-rtc items:")
        items = collection_helper.get_items(geometries=[geometry], max_items=5)

        print(f"   Found {len(items)} items")

        return items[:5]

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

    print("1. Polarization Analysis:")
    pol_info = collection_helper.get_polarization_info(items)

    for item_name, polarizations in list(pol_info.items())[:3]:
        print(f"   {item_name}: {', '.join(polarizations)}")

    print("2. Orbit Information:")
    orbit_info = collection_helper.get_orbit_info(items)

    for item_name, orbit_data in list(orbit_info.items())[:3]:
        direction = orbit_data.get("orbit_direction", "Unknown")
        orbit_num = orbit_data.get("orbit_number", "Unknown")
        print(f"   {item_name}:")
        print(f"     Orbit direction: {direction}")
        print(f"     Orbit number: {orbit_num}")

    print("3. Filtering by Polarization:")
    dual_pol_items = collection_helper.filter_by_polarization(items, ["VV", "VH"])
    vv_only_items = collection_helper.filter_by_polarization(items, ["VV"])
    hh_only_items = collection_helper.filter_by_polarization(items, ["HH"])

    print(f"   Items with both VV and VH: {len(dual_pol_items)}")
    print(f"   Items with VV: {len(vv_only_items)}")
    print(f"   Items with HH: {len(hh_only_items)}")


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
        print("1. Downloading preview images:")
        preview_downloads = collection_helper.download_assets(
            items=items[:1],
            asset_keys=["rendered_preview"],
            output_dir=output_path / "sentinel1_previews",
        )

        print(f"   Downloaded {len(preview_downloads)} preview files")
        for asset_key, file_path in preview_downloads.items():
            print(f"     {asset_key}: {file_path}")

        download_vv = input("   Download VV polarization data? (y/N): ").lower() == "y"

        if download_vv:
            print("2. Downloading VV polarization data:")
            vv_downloads = collection_helper.download_assets(
                items=items[:1],
                asset_keys=["vv"],
                output_dir=output_path / "sentinel1_vv",
            )

            print(f"   Downloaded {len(vv_downloads)} VV files")
            for asset_key, file_path in vv_downloads.items():
                print(f"     {asset_key}: {file_path}")

    except Exception as e:
        print(f"Error downloading assets: {e}")


def create_sentinel1_datacube(client, geometry):
    """Create a Sentinel-1 datacube."""
    print("Creating Sentinel-1 Datacube")
    print("-" * 35)

    try:
        s1_source = Sentinel1CollectionHelper.create_single_pol_vv(client)
        items = s1_source.get_items(geometries=[geometry], max_items=5)

        if not items:
            print("   No items found for datacube creation")
            return None
        print(f"   Using {len(items)} items for datacube")

        datacube = s1_source.create_datacube(items=items, assets=["vv"])

        print("Datacube created successfully!")
        print(f"   Shape: {datacube.shape}")
        print(f"   Bands: {datacube.bands}")
        if datacube.timestamps:
            print(f"   Time range: {datacube.timestamps[0]} to {datacube.timestamps[-1]}")

        if "vv" in datacube.bands:
            vv_data = datacube.data["vv"]
            print("   VV statistics:")
            print(f"     Mean: {float(vv_data.mean()):.6f}")
            print(f"     Std: {float(vv_data.std()):.6f}")
            print(f"     Min/Max: {float(vv_data.min()):.6f} / {float(vv_data.max()):.6f}")

        return datacube

    except Exception as e:
        print(f"Error creating datacube: {e}")
        return None


def main():
    """Main function to demonstrate Sentinel-1 collection helper capabilities."""
    try:
        print("Sentinel-1 Collection Helper Example")
        print("=" * 40)

        client = initialize_client()
        geometry = create_test_geometry()
        s1_collection_helper = demonstrate_collection_helper_creation(client)
        items = search_sentinel1_items(s1_collection_helper, geometry)
        analyze_item_metadata(s1_collection_helper, items)

        if items:
            download_demo = input("   Run download demonstration? (y/N): ").lower() == "y"
            if download_demo:
                demonstrate_asset_downloads(s1_collection_helper, items)

        datacube = create_sentinel1_datacube(client, geometry)

        print(f"Datacube: {datacube}")
        print("Sentinel-1 example completed successfully!")

    except EDSAPIError as e:
        print(f"API Error: {e}")
        print(f"   Status Code: {e.status_code}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
