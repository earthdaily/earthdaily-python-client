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

    search_stac_items(client)


if __name__ == "__main__":
    main()
