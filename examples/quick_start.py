"""
Quick Start Example
====================

This example demonstrates how to use the EarthDaily Python client v1
to search for and work with STAC items.

Features demonstrated:
- Client initialization with environment variables
- Searching for STAC items
- Basic error handling
- Asset information access

Requirements:
- Set your EDS credentials as environment variables or in a .env file:
  EDS_CLIENT_ID, EDS_SECRET, EDS_AUTH_URL, EDS_API_URL
"""

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("💡 Consider installing the utils extra to automatically load .env files:")
    print("   pip install 'earthdaily[platform,utils]'")

from earthdaily import EDSClient, EDSConfig
from earthdaily.exceptions import EDSAPIError


def initialize_client():
    """Initialize the EarthDaily API client with environment variables."""
    print("🚀 Initializing EarthDaily Client...")

    # EDSConfig will automatically read from environment variables:
    # EDS_CLIENT_ID, EDS_SECRET, EDS_AUTH_URL, EDS_API_URL
    config = EDSConfig()
    client = EDSClient(config)

    print("✅ Client initialized successfully!")
    return client


def search_stac_items(client):
    """Search for STAC items using the platform API."""
    try:
        print("\n🔍 Searching for Sentinel-2 L2A items...")

        # Search for recent Sentinel-2 items with cloud mask available
        search_result = client.platform.pystac_client.search(
            collections=["sentinel-2-l2a"],
            query={"eda:ag_cloud_mask_available": {"eq": True}},
            datetime="2024-06-01T00:00:00Z/2024-08-01T00:00:00Z",
            max_items=5,  # Limit results for demo
        )

        items = list(search_result.items())
        print(f"\n🌍 Found {len(items)} STAC items:")

        for i, item in enumerate(items, 1):
            print(f"  {i}. {item.id}")
            print(f"     Date: {item.datetime}")
            print(f"     Cloud cover: {item.properties.get('eo:cloud_cover', 'N/A')}%")
            print(f"     Assets: {len(item.assets)} available")

            # Show some key assets
            key_assets = ["red", "green", "blue", "nir", "visual", "thumbnail"]
            available_key_assets = [asset for asset in key_assets if asset in item.assets]
            if available_key_assets:
                print(f"     Key assets: {', '.join(available_key_assets)}")
            print()

        return items

    except EDSAPIError as e:
        print(f"\n❌ API error: {e}")
        print(f"   Status Code: {e.status_code}")
        print(f"   Details: {e.body}")
        return []
    except Exception as e:
        print(f"\n💥 Unexpected error while searching: {e}")
        return []


def explore_item_details(item):
    """Explore details of a STAC item."""
    print(f"\n🔍 Exploring item: {item.id}")
    print(f"   Collection: {item.collection_id}")
    print(f"   Geometry type: {item.geometry['type']}")
    print(f"   Bounding box: {item.bbox}")

    # Show properties
    print("\n   Key properties:")
    key_props = ["datetime", "eo:cloud_cover", "gsd", "platform", "constellation"]
    for prop in key_props:
        if prop in item.properties:
            print(f"     {prop}: {item.properties[prop]}")

    # Show available assets
    print(f"\n   Available assets ({len(item.assets)}):")
    for asset_name, asset in item.assets.items():
        print(f"     {asset_name}: {asset.media_type or 'Unknown type'}")
        if hasattr(asset, "extra_fields") and "gsd" in asset.extra_fields:
            print(f"       Resolution: {asset.extra_fields['gsd']}m")


def main():
    """Main function to demonstrate the quick start workflow."""
    try:
        # Initialize client
        client = initialize_client()

        # Search for items
        items = search_stac_items(client)

        if items:
            # Explore the first item in detail
            explore_item_details(items[0])

            print("\n✨ Quick start completed successfully!")
            print(f"   Found {len(items)} items to work with.")
            print("   Try the other examples to learn about:")
            print("   - Creating datacubes (datacube_example.py)")
            print("   - Downloading assets (asset_download_example.py)")
            print("   - Bulk operations (bulk_search_example.py)")
        else:
            print("\n❌ No items found. Check your search parameters and try again.")

    except Exception as e:
        print(f"\n💥 Error in main workflow: {e}")
        print("\n💡 Make sure you have set your EDS credentials as environment variables:")
        print("   EDS_CLIENT_ID, EDS_SECRET, EDS_AUTH_URL, EDS_API_URL")


if __name__ == "__main__":
    main()
