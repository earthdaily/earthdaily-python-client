"""
Quick Start Example
====================

This example demonstrates how to use the EarthDaily Python client
to fetch and process STAC items.

- Initializes an EDS client.
- Queries the legacy API.
- Searches the platform for STAC items.
- Handles API errors gracefully.

"""

from earthdaily import EDSClient, EDSConfig
from earthdaily.exceptions import EDSAPIError


def initialize_client():
    """Initialize the EarthDaily API client with default configuration."""
    config = EDSConfig()
    return EDSClient(config)


def search_stac_items(client):
    """Search for STAC items in the EarthDaily platform."""
    try:
        print("\n🔍 Searching for Venus-L2A items...")
        search_result = client.platform.pystac_client.search(
            collections=["sentinel-2-l2a"],
            query={"eda:ag_cloud_mask_available": {"eq": True}},
            datetime="2024-06-01T00:00:00Z/2024-08-01T00:00:00Z",
            max_items=50,
        )

        items = search_result.items()
        print("\n🌍 STAC items Found:")
        for item in items:
            print(item)

    except EDSAPIError as e:
        print(f"\n❌ API error: {e}\nStatus Code: {e.status_code}\nDetails: {e.body}")
    except Exception as e:
        print(f"\nUnexpected error while searching for data: {e}")


def main():
    """Main function to execute the quick start workflow."""
    print("\n🚀 Initializing EarthDaily Client...")
    client = initialize_client()
    # r = client.platform.stac_item.get_item("venus-l2a", "VENUS-XS_20171208-143816-000_L2A_25MAYO_D")
    stac_item_data = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": ["https://earthdaily-stac-extensions.s3.us-east-1.amazonaws.com/eda/v0.1.0/schema.json"],
        "id": "test-1234",
        "collection": "eda-labels-vessels",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-68.003608703613, -38.143650054932],
                    [-67.412925720215, -38.143650054932],
                    [-67.412925720215, -37.555374145508],
                    [-68.003608703613, -37.555374145508],
                    [-68.003608703613, -38.143650054932],
                ]
            ],
        },
        "bbox": [-68.003608703613, -38.143650054932, -67.412925720215, -37.555374145508],
        "properties": {
            "gsd": 5,
            "title": "test-1234",
            "license": "CC-BY-NC-4.0",
            "mission": "venus",
            "platform": "VENUS",
            "proj:epsg": 32719,
            "providers": [
                {"name": "Theia", "roles": ["licensor", "producer", "processor"]},
                {"url": "https://earthdaily.com", "name": "EarthDaily Analytics", "roles": ["processor", "host"]},
            ],
            "view:azimuth": 191.617727,
            "constellation": "VENUS",
            "eda:water_cover": 0,
            "eda:product_type": "REFLECTANCE",
            "processing:level": "L2A",
            "theia:product_id": "VENUS-XS_20171208-143816-000_L2A_25MAYO_C_V3-0",
            "view:sun_azimuth": 63.9400504497,
            "theia:sensor_mode": "XS",
            "theia:source_uuid": "fc6ced30-0294-5afe-9699-a9c4705c03a0",
            "sat:absolute_orbit": 1863,
            "view:sun_elevation": 63.0446423284,
            "view:incidence_angle": 26.070811499999998,
            "theia:product_version": "3.0",
            "theia:publication_date": "2022-07-12T17:06:12.613000Z",
            "eo:cloud_cover": 0,
            "start_datetime": "2017-12-08T14:38:16.000000Z",
            "end_datetime": "2017-12-08T14:38:16.000000Z",
            "created": "2023-03-29T02:38:21.634449Z",
            "updated": "2023-10-04T21:54:26.373346Z",
            "datetime": "2017-12-08T14:38:16.000000Z",
        },
        "links": [],
        "assets": {},
    }
    try:
        r = client.platform.stac_item.create_item("eda-labels-vessels", stac_item_data)
        print(f"\n📦 Retrieved STAC Item: {r}")
    except EDSAPIError as e:
        print(f"\n❌ API error: {e}\nStatus Code: {e.status_code}\nDetails: {e.body}")

    r = client.platform.stac_item.get_item("eda-labels-vessels", "test-1234")
    print(f"\n📦 Retrieved STAC Item: {r}")

    # Update the item
    stac_item_data["properties"]["eda:water_cover"] = 0.5
    r = client.platform.stac_item.update_item("eda-labels-vessels", "test-1234", stac_item_data)
    print(f"\n📦 Updated STAC Item: {r}")

    client.platform.stac_item.delete_item("eda-labels-vessels", "test-1234")
    print("\n📦 Deleted STAC Item")

    # search_stac_items(client)


if __name__ == "__main__":
    main()
