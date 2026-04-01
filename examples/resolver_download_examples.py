"""
Resolver Download Examples
==========================

Demonstrates downloading assets that go through each download resolver:

1. EarthDaily API Resolver  -- sentinel-2-l2a (presigned *.earthdaily.com URLs)
2. S3 Resolver              -- sentinel-2-l1c (s3:// and *.amazonaws.com URLs)
3. EarthData Resolver       -- nasa:hls:l30:v1 (NASA EarthData URLs, requires token)
4. EUMETSAT Resolver        -- msg-seviri (api.eumetsat.int URLs, requires credentials)

Requirements:
- Set your EDS credentials as environment variables or in a .env file
- Install with platform + download support: pip install 'earthdaily[platform,download]'
- For EarthData: set EARTHDATA_TOKEN env var (or use internal client)
- For EUMETSAT: set EUMETSAT_CLIENT_ID and EUMETSAT_CLIENT_SECRET env vars (or use internal client)
"""

from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from earthdaily import EDSClient, EDSConfig
from earthdaily._eds_config import AssetAccessMode
from earthdaily.exceptions import EDSAPIError

OUTPUT_BASE = Path.home() / "Downloads" / "earthdaily_resolver_demos"


def make_client() -> EDSClient:
    return EDSClient(EDSConfig())


def inspect_asset_urls(item, max_show: int = 5) -> None:
    """Print the first few asset hrefs so we can see which resolver will handle them."""
    for i, (key, asset) in enumerate(item.assets.items()):
        if i >= max_show:
            print(f"   ... and {len(item.assets) - max_show} more assets")
            break
        href = asset.href
        parsed = urlparse(href)
        print(f"   {key}: {parsed.scheme}://{parsed.netloc}/...  ({asset.media_type or 'unknown type'})")


def demo_earthdaily_api_resolver():
    """Resolver: EarthDailyAPIResolver -- standard presigned URLs from *.earthdaily.com."""
    print("=" * 70)
    print("1. EarthDaily API Resolver  (sentinel-2-l2a)")
    print("=" * 70)

    client = make_client()

    try:
        results = client.platform.pystac_client.search(
            collections=["sentinel-2-l2a"],
            datetime="2024-06-01/2024-06-30",
            query={"eo:cloud_cover": {"lt": 10}},
            max_items=1,
        )
        items = list(results.items())
    except (EDSAPIError, Exception) as e:
        print(f"   Search failed: {e}")
        return

    if not items:
        print("   No items found.")
        return

    item = items[0]
    print(f"   Item: {item.id}")
    inspect_asset_urls(item)

    asset_key = next((k for k in ["thumbnail", "visual"] if k in item.assets), None)
    if not asset_key:
        print("   No lightweight asset to download, skipping.")
        return

    output_dir = OUTPUT_BASE / "earthdaily_api"
    try:
        result = client.platform.stac_item.download_assets(
            item=item,
            asset_keys=[asset_key],
            output_dir=output_dir,
        )
        for key, path in result.items():
            print(f"   Downloaded {key} -> {path}  ({path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"   Download failed: {e}")


def demo_s3_resolver():
    """Resolver: S3Resolver -- s3:// or *.amazonaws.com URLs.

    Using AssetAccessMode.RAW gives us the native s3:// hrefs directly,
    which the S3Resolver picks up and downloads via boto3.
    """
    print("\n" + "=" * 70)
    print("2. S3 Resolver  (sentinel-2-l1c)")
    print("=" * 70)

    client = EDSClient(EDSConfig(asset_access_mode=AssetAccessMode.RAW))

    try:
        results = client.platform.pystac_client.search(
            collections=["sentinel-2-l1c"],
            datetime="2024-06-01/2024-06-30",
            query={"eo:cloud_cover": {"lt": 10}},
            max_items=1,
        )
        items = list(results.items())
    except (EDSAPIError, Exception) as e:
        print(f"   Search failed: {e}")
        return

    if not items:
        print("   No items found.")
        return

    item = items[0]
    print(f"   Item: {item.id}")
    inspect_asset_urls(item)

    asset_key = next(
        (k for k, a in item.assets.items() if a.href.startswith("s3://") and not a.href.endswith("/")),
        None,
    )
    if not asset_key:
        print("   No downloadable S3 asset found, skipping.")
        return

    print(f"   Downloading asset: {asset_key}")

    output_dir = OUTPUT_BASE / "s3"
    try:
        result = client.platform.stac_item.download_assets(
            item=item,
            asset_keys=[asset_key],
            output_dir=output_dir,
            href_type=None,
        )
        for key, path in result.items():
            print(f"   Downloaded {key} -> {path}  ({path.stat().st_size / 1024:.1f} KB)")
    except ImportError:
        print("   boto3 not installed -- install with: pip install 'earthdaily[download]'")
    except Exception as e:
        print(f"   Download failed: {e}")


def demo_earthdata_resolver():
    """Resolver: EarthDataResolver -- NASA EarthData URLs.

    nasa:hls:l30:v1 items link to NASA EarthData domains.
    Requires EARTHDATA_TOKEN env var or internal client configuration.
    """
    print("\n" + "=" * 70)
    print("3. EarthData Resolver  (nasa:hls:l30:v1)")
    print("   Requires: EARTHDATA_TOKEN env var")
    print("=" * 70)

    client = make_client()

    try:
        results = client.platform.pystac_client.search(
            collections=["nasa:hls:l30:v1"],
            max_items=1,
        )
        items = list(results.items())
    except (EDSAPIError, Exception) as e:
        print(f"   Search failed: {e}")
        return

    if not items:
        print("   No items found.")
        return

    item = items[0]
    print(f"   Item: {item.id}")
    inspect_asset_urls(item)

    asset_key = next(iter(item.assets), None)
    if not asset_key:
        print("   No assets found, skipping.")
        return

    output_dir = OUTPUT_BASE / "earthdata"
    try:
        result = client.platform.stac_item.download_assets(
            item=item,
            asset_keys=[asset_key],
            output_dir=output_dir,
            href_type=None,
        )
        for key, path in result.items():
            print(f"   Downloaded {key} -> {path}  ({path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"   Download failed: {e}")
        print("   Hint: set EARTHDATA_TOKEN or configure the internal client.")


def demo_eumetsat_resolver():
    """Resolver: EUMETSATResolver -- api.eumetsat.int URLs.

    msg-seviri items link to EUMETSAT download endpoints.
    Using AssetAccessMode.RAW to get the native EUMETSAT URLs.
    Requires EUMETSAT_CLIENT_ID and EUMETSAT_CLIENT_SECRET env vars or
    internal client configuration.
    """
    print("\n" + "=" * 70)
    print("4. EUMETSAT Resolver  (msg-seviri)")
    print("   Requires: EUMETSAT_CLIENT_ID + EUMETSAT_CLIENT_SECRET env vars")
    print("=" * 70)

    client = EDSClient(EDSConfig(asset_access_mode=AssetAccessMode.RAW))

    try:
        results = client.platform.pystac_client.search(
            collections=["msg-seviri"],
            max_items=1,
        )
        items = list(results.items())
    except (EDSAPIError, Exception) as e:
        print(f"   Search failed: {e}")
        return

    if not items:
        print("   No items found.")
        return

    item = items[0]
    print(f"   Item: {item.id}")
    inspect_asset_urls(item)

    asset_key = next(iter(item.assets), None)
    if not asset_key:
        print("   No assets found, skipping.")
        return

    output_dir = OUTPUT_BASE / "eumetsat"
    try:
        result = client.platform.stac_item.download_assets(
            item=item,
            asset_keys=[asset_key],
            output_dir=output_dir,
            href_type=None,
        )
        for key, path in result.items():
            print(f"   Downloaded {key} -> {path}  ({path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"   Download failed: {e}")
        print("   Hint: set EUMETSAT_CLIENT_ID and EUMETSAT_CLIENT_SECRET,")
        print("         or configure the internal client.")


DEMOS = {
    "1": ("EarthDaily API (sentinel-2-l2a)", demo_earthdaily_api_resolver),
    "2": ("S3 (sentinel-2-l1c)", demo_s3_resolver),
    "3": ("EarthData / NASA (nasa:hls:l30:v1)", demo_earthdata_resolver),
    "4": ("EUMETSAT (msg-seviri)", demo_eumetsat_resolver),
}


def main():
    print("Resolver Download Examples")
    print("=" * 70)
    print("\nEach demo exercises a different download resolver.\n")

    for num, (label, _) in DEMOS.items():
        print(f"  {num}. {label}")
    print("  a. Run all demos")

    choice = input("\nChoice: ").strip().lower()

    if choice == "a":
        for _, fn in DEMOS.values():
            fn()
    elif choice in DEMOS:
        DEMOS[choice][1]()
    else:
        print("Invalid choice -- running all demos.")
        for _, fn in DEMOS.values():
            fn()

    print(f"\nFiles saved under: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
