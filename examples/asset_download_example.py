"""
Asset Download Example
======================

This example demonstrates how to use the EarthDaily Python client v1
to download assets from STAC items.

Features demonstrated:
- Searching for STAC items
- Downloading specific assets
- Working with different asset access modes
- Progress monitoring during downloads
- Organizing downloaded files

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


def initialize_client(asset_access_mode="presigned-urls"):
    """Initialize the EarthDaily API client with specified asset access mode."""
    print(f"🚀 Initializing EarthDaily Client (Asset mode: {asset_access_mode})...")

    # Map string to enum
    mode_mapping = {"presigned-urls": AssetAccessMode.PRESIGNED_URLS, "proxy-urls": AssetAccessMode.PROXY_URLS}

    config = EDSConfig(asset_access_mode=mode_mapping.get(asset_access_mode, AssetAccessMode.PRESIGNED_URLS))
    client = EDSClient(config)
    print("✅ Client initialized successfully!")
    return client


def search_for_items(client, collection="sentinel-2-l2a", max_items=3):
    """Search for STAC items to download."""
    print(f"\n🔍 Searching for {collection} items...")

    try:
        search_result = client.platform.pystac_client.search(
            collections=[collection],
            datetime="2024-06-01T00:00:00Z/2024-06-30T23:59:59Z",
            query={"eo:cloud_cover": {"lt": 10}},  # Low cloud cover
            max_items=max_items,
        )

        items = list(search_result.items())
        print(f"✅ Found {len(items)} items with <10% cloud cover")

        for i, item in enumerate(items, 1):
            print(f"   {i}. {item.id}")
            print(f"      Date: {item.datetime}")
            print(f"      Cloud cover: {item.properties.get('eo:cloud_cover', 'N/A')}%")
            print(f"      Available assets: {len(item.assets)}")

        return items

    except EDSAPIError as e:
        print(f"❌ Error searching for items: {e}")
        return []
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return []


def show_available_assets(item):
    """Display available assets for a STAC item."""
    print(f"\n📦 Available assets for {item.id}:")

    # Group assets by type
    visual_assets = []
    spectral_assets = []
    metadata_assets = []

    for asset_name, asset in item.assets.items():
        media_type = getattr(asset, "media_type", "Unknown")
        if any(keyword in asset_name.lower() for keyword in ["visual", "thumbnail", "overview"]):
            visual_assets.append((asset_name, media_type))
        elif any(keyword in asset_name.lower() for keyword in ["red", "green", "blue", "nir", "swir"]):
            spectral_assets.append((asset_name, media_type))
        else:
            metadata_assets.append((asset_name, media_type))

    if visual_assets:
        print("   📸 Visual assets:")
        for name, media_type in visual_assets:
            print(f"      - {name} ({media_type})")

    if spectral_assets:
        print("   🌈 Spectral assets:")
        for name, media_type in spectral_assets:
            print(f"      - {name} ({media_type})")

    if metadata_assets:
        print("   📄 Metadata assets:")
        for name, media_type in metadata_assets[:5]:  # Show first 5
            print(f"      - {name} ({media_type})")
        if len(metadata_assets) > 5:
            print(f"      ... and {len(metadata_assets) - 5} more metadata assets")


def download_specific_assets(client, item, asset_keys, output_dir):
    """Download specific assets from a STAC item."""
    print(f"\n Downloading assets: {asset_keys}")
    print(f"   Item: {item.id}")
    print(f"   Output directory: {output_dir}")

    try:
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Download assets
        client.platform.stac_item.download_assets(
            item=item,
            asset_keys=asset_keys,
            output_dir=output_path,
            max_workers=4,  # Parallel downloads
        )

        print("✅ Assets downloaded successfully!")

        # List downloaded files
        downloaded_files = list(output_path.glob("*"))
        if downloaded_files:
            print("   Downloaded files:")
            for file in downloaded_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"     - {file.name} ({size_mb:.1f} MB)")

        return True

    except Exception as e:
        print(f"❌ Error downloading assets: {e}")
        return False


def download_by_item_string(client, item_string, asset_keys, output_dir):
    """Download assets using item string (collection/item_id format)."""
    print(f"\n📥 Downloading from item string: {item_string}")

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        client.platform.stac_item.download_assets(
            item=item_string,
            asset_keys=asset_keys,
            output_dir=output_path,
            href_type="alternate.download.href",  # Use alternate download links
            max_workers=2,
        )

        print("✅ Assets downloaded successfully using item string!")
        return True

    except Exception as e:
        print(f"❌ Error downloading with item string: {e}")
        return False


def demo_basic_download():
    """Demonstrate basic asset downloading."""
    print("=" * 60)
    print("📥 DEMO: Basic Asset Download")
    print("=" * 60)

    # Initialize client
    client = initialize_client("presigned-urls")

    # Search for items
    items = search_for_items(client, "sentinel-2-l2a", 1)

    if not items:
        print("❌ No items found for download demo")
        return

    item = items[0]

    # Show available assets
    show_available_assets(item)

    # Download visual assets
    visual_assets = ["visual", "thumbnail"]
    available_visual = [asset for asset in visual_assets if asset in item.assets]

    if available_visual:
        output_dir = Path.home() / "Downloads" / "earthdaily_assets" / "basic_demo"
        success = download_specific_assets(client, item, available_visual, output_dir)

        if success:
            print("\n✨ Basic download demo completed!")
            print(f"💡 Check downloaded files in: {output_dir}")
    else:
        print("❌ No visual assets available for download")


def demo_advanced_download():
    """Demonstrate advanced asset downloading with different configurations."""
    print("\n" + "=" * 60)
    print("🛠️  DEMO: Advanced Asset Download")
    print("=" * 60)

    print("\nThis demo shows different download configurations:")
    print("1. Proxy URLs vs Presigned URLs")
    print("2. Different asset types")
    print("3. Parallel downloads")

    # Compare different access modes
    for mode in ["presigned-urls", "proxy-urls"]:
        print(f"\n--- Testing {mode} ---")

        client = initialize_client(mode)
        items = search_for_items(client, "sentinel-2-l2a", 1)

        if items:
            item = items[0]

            # Download spectral assets
            spectral_assets = ["red", "green", "blue"]
            available_spectral = [asset for asset in spectral_assets if asset in item.assets]

            if available_spectral:
                output_dir = Path.home() / "Downloads" / "earthdaily_assets" / f"advanced_demo_{mode}"
                download_specific_assets(client, item, available_spectral[:2], output_dir)


def demo_collection_download():
    """Demonstrate downloading from different collections."""
    print("\n" + "=" * 60)
    print("🌍 DEMO: Multi-Collection Download")
    print("=" * 60)

    collections_to_try = ["sentinel-2-l2a", "landsat-c2l2-sr"]
    client = initialize_client()

    for collection in collections_to_try:
        print(f"\n--- Processing {collection} ---")

        items = search_for_items(client, collection, 1)

        if items:
            item = items[0]

            # Try to download thumbnail or visual asset
            preferred_assets = ["thumbnail", "visual", "overview"]
            available_assets = [asset for asset in preferred_assets if asset in item.assets]

            if available_assets:
                output_dir = Path.home() / "Downloads" / "earthdaily_assets" / f"collection_demo_{collection}"
                download_specific_assets(client, item, available_assets[:1], output_dir)
            else:
                print(f"   No preferred assets available for {collection}")


def main():
    """Main function to demonstrate asset download workflows."""
    try:
        print("📥 EarthDaily Asset Download Examples")
        print("=" * 60)
        print("\nThis example demonstrates various asset download methods.")
        print("Choose a demo to run:\n")
        print("1. Basic asset download (visual assets)")
        print("2. Advanced download configurations")
        print("3. Multi-collection download")
        print("4. Run all demos")

        choice = input("\nEnter your choice (1/2/3/4): ").strip()

        if choice == "1":
            demo_basic_download()
        elif choice == "2":
            demo_advanced_download()
        elif choice == "3":
            demo_collection_download()
        elif choice == "4":
            demo_basic_download()
            demo_advanced_download()
            demo_collection_download()
        else:
            print("❌ Invalid choice. Running basic demo by default...")
            demo_basic_download()

        print("\n🎉 Download examples completed!")
        print("💡 Downloaded files are organized in your Downloads/earthdaily_assets/ folder")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("💡 Make sure to install with platform support:")
        print("   pip install 'earthdaily[platform]'")
    except Exception as e:
        print(f"\n💥 Unexpected error in main: {e}")
        print("\n💡 Make sure you have set your EDS credentials as environment variables:")
        print("   EDS_CLIENT_ID, EDS_SECRET, EDS_AUTH_URL, EDS_API_URL")


if __name__ == "__main__":
    main()
