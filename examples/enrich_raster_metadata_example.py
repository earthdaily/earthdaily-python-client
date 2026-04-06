"""
Enrich Raster Metadata Example
================================

Demonstrates how to enrich STAC items with projection and raster band
metadata using ``client.datacube.enrich_raster_metadata()``.

This is useful when a collection's STAC items lack ``proj:`` or
``raster:bands`` fields and you need them for downstream analysis
(e.g., datacube creation, reprojection, spatial alignment).

The example uses the ``sentinel-2-l2a`` collection.

Requirements:
- pip install 'earthdaily[platform,datacube]'
- Set EDS credentials as environment variables
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from earthdaily import EDSClient, EDSConfig


def _print_asset_metadata(item, label):
    """Print projection and raster metadata for each tiff asset."""
    print(f"\n--- {label}: {item.id} ---")
    print(f"  properties.gsd: {item.properties.get('gsd')}")
    print(f"  properties.proj:code: {item.properties.get('proj:code')}")

    for name, asset in item.assets.items():
        if not asset.media_type or not asset.media_type.startswith("image/tiff"):
            continue
        ef = asset.extra_fields
        print(f"  asset '{name}':")
        print(f"    proj:code      = {ef.get('proj:code')}")
        print(f"    proj:shape     = {ef.get('proj:shape')}")
        print(f"    proj:transform = {ef.get('proj:transform')}")
        print(f"    gsd            = {ef.get('gsd')}")
        bands = ef.get("raster:bands")
        if bands:
            print(f"    raster:bands   = {len(bands)} band(s)")
            for i, b in enumerate(bands):
                print(f"      [{i}] data_type={b.get('data_type')}, nodata={b.get('nodata')}")
        else:
            print("    raster:bands   = None")


def main():
    config = EDSConfig()
    client = EDSClient(config)

    print("Searching sentinel-2-l2a ...")
    items = client.platform.search(
        collections=["sentinel-2-l2a"],
        max_items=2,
    )
    print(f"Found {len(items)} item(s)")

    if not items:
        print("No items found.")
        return

    for item in items:
        _print_asset_metadata(item, "BEFORE enrichment")

    print("\n\nEnriching raster metadata ...")
    enriched = [client.datacube.enrich_raster_metadata(item) for item in items]

    for item in enriched:
        _print_asset_metadata(item, "AFTER enrichment")

    print("\nDone.")


if __name__ == "__main__":
    main()
