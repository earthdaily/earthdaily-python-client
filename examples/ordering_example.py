"""
EarthDaily Ordering Examples

This example demonstrates how to use the EarthDaily ordering service to create orders
and check order status.

Setup:
    This example requires both ordering and platform dependencies. Install with:

    pip install 'earthdaily[ordering,platform]'

    Note: Installing only 'earthdaily[ordering]' will not be sufficient as some
    examples use client.platform which requires platform dependencies.
"""

from earthdaily import EDSClient, EDSConfig
from earthdaily.ordering import EdcLineItem, EdcOrder, EdcProductType


def order_with_strings(client: EDSClient):
    print("=" * 80)
    print("Example 1: Order using item ID strings with EdcProductType enum")
    print("=" * 80)

    item_ids = [
        "20221215T054109.675400Z_EarthDaily03VNIR_Catalog_L1C_TOA_4dd3e27c",
        "20221215T054108.961449Z_EarthDaily03VNIR_Catalog_L1C_TOA_1507bf9d",
    ]

    order: EdcOrder = client.ordering.edc.create(items=item_ids, product_type=EdcProductType.VISUAL_RGB)

    print(f"EDC Order ID: {order.id}")
    print(f"Tracking ID: {order.tracking_id}")
    print(f"Submission Date: {order.submission_date}")
    print("\nLine Items (from creation):")
    for line_item in order.line_items:
        print(f"  - Order Name: {line_item.order_name}")
        print(f"    State: {line_item.state}")
        print(f"    Input UUID: {line_item.input_uuid}")


def order_with_stac_item(client: EDSClient):
    print("=" * 80)
    print("Example 2: Order using STAC Item with product_type as string")
    print("=" * 80)

    target_item_id = "20221215T054109.677024Z_EarthDaily03VNIR_Catalog_L1C_TOA_d0a8ccf9"

    search = client.platform.pystac_client.search(
        collections=["simedc-vnir-catalog"], ids=[target_item_id], max_items=1
    )

    items = list(search.items())

    if not items:
        print("No items found in catalog search")
        return None

    print(f"Found {len(items)} item(s) from catalog search")
    print(f"Item ID: {items[0].id}")

    order: EdcOrder = client.ordering.edc.create(items=[items[0]], product_type="visual_rgb")

    print(f"\nEDC Order ID: {order.id}")
    print(f"Tracking ID: {order.tracking_id}")
    print(f"Submission Date: {order.submission_date}")
    print("\nLine Items (from creation):")
    for line_item in order.line_items:
        print(f"  - Order Name: {line_item.order_name}")
        print(f"    State: {line_item.state}")
        print(f"    Input UUID: {line_item.input_uuid}")


def get_order_status(client: EDSClient):
    print("=" * 80)
    print("Example 3: Get order status using order ID")
    print("=" * 80)

    order_id = "2EgfLKfmCwXEPCQ4qKdJ8g"

    order: EdcOrder = client.ordering.edc.get(order_id)
    print(f"\nEDC Order ID: {order.id}")
    print(f"Submission Date: {order.submission_date}")

    line_items: list[EdcLineItem] = client.ordering.edc.get_line_items(order_id)
    print(f"\nLine Items ({len(line_items)} total):")
    for line_item in line_items:
        print(f"  - Order Name: {line_item.order_name}")
        print(f"    State: {line_item.state}")
        print(f"    Input UUID: {line_item.input_uuid}")
        if line_item.output_uuid:
            print(f"    Output UUID: {line_item.output_uuid}")
        if line_item.processed_products:
            for product in line_item.processed_products:
                print(f"    Collection: {product.collection}")
                print(f"    Product State: {product.state}")


def main():
    config = EDSConfig()
    client = EDSClient(config)

    print("\nEarthDaily Ordering Examples")
    print("=" * 80)
    print("1. Order using item ID strings with EdcProductType enum")
    print("2. Order using STAC Item with product_type as string")
    print("3. Get order status using order ID")
    print("=" * 80)

    choice = input("\nSelect an example (1-3): ").strip()

    if choice == "1":
        order_with_strings(client)
    elif choice == "2":
        order_with_stac_item(client)
    elif choice == "3":
        get_order_status(client)
    else:
        print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
