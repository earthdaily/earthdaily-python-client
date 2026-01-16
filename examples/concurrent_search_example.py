"""
Concurrent Search Example
=========================

Demonstrates platform.search() with concurrent execution for improved performance
on large date ranges.

Requirements:
- EDS credentials as environment variables or in a .env file
- Install with platform support: pip install 'earthdaily[platform]'
"""

import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from earthdaily import EDSClient, EDSConfig


def run_search(name, search_func):
    """Run a search and measure performance."""
    start_time = time.perf_counter()
    items = search_func()
    elapsed = time.perf_counter() - start_time
    return name, len(items), elapsed


def main():
    config = EDSConfig()
    client = EDSClient(config)

    collections = ["sentinel-2-l2a"]
    datetime_range = "2023-01-01/2024-12-31"
    bbox = [-122.5, 37.5, -121.5, 38.5]
    max_items = 1000
    limit = 100

    print("Concurrent Search Performance Comparison")
    print("=" * 70)
    print(f"Collection:  {collections[0]}")
    print(f"Date range:  {datetime_range} (2 years)")
    print(f"Bbox:        {bbox}")
    print(f"Max items:   {max_items} (standard search only)")
    print(f"Page limit:  {limit}")
    print("=" * 70)
    print()

    results = []

    results.append(
        run_search(
            "pystac_client.search()",
            lambda: client.platform.pystac_client.search(
                collections=collections,
                datetime=datetime_range,
                bbox=bbox,
                max_items=max_items,
                limit=limit,
            ).item_collection(),
        )
    )

    results.append(
        run_search(
            "platform.search()",
            lambda: client.platform.search(
                collections=collections,
                datetime=datetime_range,
                bbox=bbox,
                max_items=max_items,
                limit=limit,
            ),
        )
    )

    results.append(
        run_search(
            "platform.search(days_per_chunk='auto')",
            lambda: client.platform.search(
                collections=collections,
                datetime=datetime_range,
                bbox=bbox,
                days_per_chunk="auto",
                limit=limit,
            ),
        )
    )

    results.append(
        run_search(
            "platform.search(days_per_chunk=7)",
            lambda: client.platform.search(
                collections=collections,
                datetime=datetime_range,
                bbox=bbox,
                days_per_chunk=7,
                limit=limit,
            ),
        )
    )

    results.append(
        run_search(
            "platform.search(days_per_chunk=7, max_items_per_chunk=200)",
            lambda: client.platform.search(
                collections=collections,
                datetime=datetime_range,
                bbox=bbox,
                days_per_chunk=7,
                max_items_per_chunk=200,
                limit=limit,
            ),
        )
    )

    print(f"{'Method':<55} {'Items':>8} {'Time':>10}")
    print("-" * 75)
    for name, count, elapsed in results:
        print(f"{name:<55} {count:>8} {elapsed:>9.2f}s")


if __name__ == "__main__":
    main()
