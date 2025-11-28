"""
Proxy URL Download Example
==========================

This example demonstrates how to use proxy URLs for downloading assets from STAC items.

Proxy URLs are non-expiring URLs that require authentication headers and redirect (307)
to the final download URL. They are only available for specific collections.

Proxy URL mode is automatically enabled when using AssetAccessMode.PROXY_URLS in your config.

Features demonstrated:
- Configuring proxy URL mode via AssetAccessMode.PROXY_URLS
- Automatic authentication with proxy URLs
- Automatic handling of redirects for proxy URLs

Requirements:
- Set your EDS credentials as environment variables or in a .env file
- Install with platform support: pip install 'earthdaily[platform]'
"""

from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("💡 Consider installing the utils extra to automatically load .env files:")
    print("   pip install 'earthdaily[platform,utils]'")

from earthdaily import EDSClient, EDSConfig
from earthdaily._eds_config import AssetAccessMode
from earthdaily.exceptions import EDSAPIError


def initialize_client():
    """Initialize the EarthDaily API client with proxy URLs enabled."""
    print("🚀 Initializing EarthDaily Client for Proxy URLs...")

    config = EDSConfig(asset_access_mode=AssetAccessMode.PROXY_URLS)
    client = EDSClient(config)
    print("✅ Client initialized successfully with proxy URL support!")
    return client


def search_for_proxy_items(client, max_items=2):
    """Search for STAC items in a collection that supports proxy URLs."""
    collection = "ai-ready-mosaics-sample"
    print(f"\n🔍 Searching for {collection} items (proxy URL enabled collection)...")

    try:
        search_result = client.platform.pystac_client.search(
            collections=[collection],
            max_items=max_items,
        )

        items = list(search_result.items())
        print(f"✅ Found {len(items)} items")

        for i, item in enumerate(items, 1):
            print(f"   {i}. {item.id}")
            print(f"      Date: {item.datetime}")
            print(f"      Available assets: {len(item.assets)}")

        return items

    except EDSAPIError as e:
        print(f"❌ Error searching for items: {e}")
        return []
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return []


def download_with_proxy_urls(client, item, asset_keys, output_dir):
    """Download assets using proxy URLs."""
    print("\n🔄 Downloading with Proxy URLs...")
    print(f"   Assets: {asset_keys}")
    print(f"   Output: {output_dir}")

    try:
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Download using proxy URLs (automatically detected from config)
        result = client.platform.stac_item.download_assets(
            item=item,
            asset_keys=asset_keys,
            output_dir=output_path,
            max_workers=1,
        )

        print("✅ Proxy URL download completed!")

        # Show downloaded files
        if result:
            print("   Downloaded files:")
            for asset_key, file_path in result.items():
                if file_path and file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"     - {asset_key}: {file_path.name} ({size_mb:.1f} MB)")

        return True

    except Exception as e:
        print(f"❌ Proxy URL download failed: {e}")
        return False


def main():
    """Main function demonstrating proxy URL downloads."""
    try:
        print("🔄 EarthDaily Proxy URL Download Example")
        print("=" * 60)
        print("\nThis example demonstrates downloading assets using proxy URLs.")
        print("Proxy URLs are automatically enabled via AssetAccessMode.PROXY_URLS config.")
        print("They are non-expiring but require authentication headers.")
        print("Proxy URLs are only available for specific collections.")

        # Initialize client
        client = initialize_client()

        # Search for items in proxy-enabled collection
        items = search_for_proxy_items(client, max_items=1)

        if not items:
            print("❌ No items found in proxy-enabled collection")
            return

        item = items[0]

        # Show item information
        print("\n📋 Item Information:")
        print(f"   Item ID: {item.id}")
        print(f"   Date: {item.datetime}")
        asset_names = list(item.assets.keys())[:5]
        print(f"   Available assets: {asset_names}")

        # Get available assets (use first few for demo)
        available_assets = list(item.assets.keys())[:2]

        if not available_assets:
            print("❌ No assets available for download")
            return

        print(f"\n🎯 Downloading assets using proxy URLs: {available_assets}")

        # Download using proxy URLs
        output_dir = Path.home() / "Downloads" / "earthdaily_proxy_demo"
        download_with_proxy_urls(client, item, available_assets, output_dir)

        print("\n🎉 Proxy URL download demonstration completed!")
        print(f"💡 Files saved to: {output_dir}")
        print("\n📋 About Proxy URLs:")
        print("   • Non-expiring URLs that require authentication headers")
        print("   • Redirect (307) to the final download URL")
        print("   • Only available for specific collections like 'ai-ready-mosaics-sample'")
        print("   • Automatically enabled when using AssetAccessMode.PROXY_URLS config")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("💡 Make sure to install with platform support:")
        print("   pip install 'earthdaily[platform]'")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("\n💡 Make sure you have set your EDS credentials as environment variables:")
        print("   EDS_CLIENT_ID, EDS_SECRET, EDS_AUTH_URL, EDS_API_URL")


if __name__ == "__main__":
    main()
