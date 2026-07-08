"""
Platform Asset Cookie Example
=============================

Demonstrates how to use the platform asset cookie with proxy URLs so GDAL
and rasterio can open authenticated assets without an ``Authorization``
header.

When ``asset_access_mode="proxy-urls"``, assets are served through
authenticated proxy URLs. GDAL cannot send the EDS bearer token, so the
client can fetch a short-lived cookie and write it to a Netscape-format
file that is passed through ``GDAL_HTTP_COOKIEFILE``.

``EDSClient`` is used as a context manager so the cookie file and the
optional background refresh thread are cleaned up on exit.

The example uses the ``ai-ready-mosaics-sample`` collection.

Requirements:
- pip install 'earthdaily[platform]'
- rasterio is required to open assets via GDAL
- Set EDS credentials as environment variables
"""

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import rasterio

from earthdaily import EDSClient, EDSConfig
from earthdaily._eds_config import AssetAccessMode


def main():
    config = EDSConfig(
        asset_access_mode=AssetAccessMode.PROXY_URLS,
        platform_cookie_enabled=True,
        platform_cookie_auto_refresh=True,
    )

    cookie_path = "/tmp/earthdaily-platform-cookies.txt"

    with EDSClient(config) as client:
        cookie_file = client.get_platform_cookie_file(path=Path(cookie_path))
        print(f"Cookie file: {cookie_file}")

        print("Searching ai-ready-mosaics-sample ...")
        search_result = client.platform.pystac_client.search(
            collections=["ai-ready-mosaics-sample"],
            max_items=1,
        )
        items = list(search_result.items())
        print(f"Found {len(items)} item(s)")

        if not items:
            print("No items found.")
            return

        item = items[0]
        asset_key = next(iter(item.assets))
        asset = item.assets[asset_key]
        asset_href = asset.extra_fields.get("alternate", {}).get("download", {}).get("href", asset.href)
        print(f"Opening asset '{asset_key}': {asset_href}")

        with rasterio.Env(
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_HTTP_COOKIEFILE=str(cookie_file),
        ):
            with rasterio.open(asset_href) as src:
                print(f"Raster: {src.width}x{src.height}, {src.count} band(s), crs={src.crs}")

    print("Done.")


if __name__ == "__main__":
    main()
