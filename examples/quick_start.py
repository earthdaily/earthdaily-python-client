"""
Quick Start Example
====================

This example demonstrates how to use the EarthDaily Python client
to fetch and process STAC items.

- Initializes an EDS client.
- Queries the agriculture API.
- Searches the platform for STAC items.
- Handles API errors gracefully.

"""

from earthdaily import EDSClient, EDSConfig
from earthdaily.exceptions import EDSAPIError


def initialize_client():
    """Initialize the EarthDaily API client with default configuration."""
    config = EDSConfig()
    return EDSClient(config)


def explore_agriculture(client):
    """Fetch agricultural data using the EarthDaily client."""
    try:
        response = client.agriculture.explore()
        print("\n✅ Agriculture Explore Response:")
        print(response)
    except EDSAPIError as e:
        print(f"\n❌ API error: {e}\nStatus Code: {e.status_code}\nDetails: {e.body}")


def search_stac_items(client):
    """Search for STAC items in the EarthDaily platform."""
    try:
        print("\n🔍 Searching for Venus-L2A items...")
        search_result = client.platform.pystac_client.search(collections=["venus-l2a"], limit=10)

        items = search_result.items()
        print("\n🌍 STAC items Found:")
        for item in items:
            print(item)
            break  # Print only the first item for brevity

    except EDSAPIError as e:
        print(f"\n❌ API error: {e}\nStatus Code: {e.status_code}\nDetails: {e.body}")
    except Exception as e:
        print(f"\nUnexpected error while searching for data: {e}")


def main():
    """Main function to execute the quick start workflow."""
    print("\n🚀 Initializing EarthDaily Client...")
    client = initialize_client()

    explore_agriculture(client)
    search_stac_items(client)


if __name__ == "__main__":
    main()
