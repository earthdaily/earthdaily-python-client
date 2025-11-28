# Ordering Module

The ordering module enables you to request processed products from the EarthDaily platform.
You can order products by specifying catalog items (STAC items or item IDs) and a product type.
The service handles order creation, status tracking, and provides detailed information about each line item in your order.

## Features

- Create product orders from STAC items or item IDs
- Track order status and monitor processing progress
- Retrieve detailed line item information including processing states
- Support for both enum and string product types

## Usage

### Basic Usage

```python
from earthdaily import EDSClient, EDSConfig
from earthdaily.ordering import EdcProductType

config = EDSConfig()
client = EDSClient(config)

# Create order from item IDs
order = client.ordering.edc.create(
    items=["item_id_1", "item_id_2"],
    product_type=EdcProductType.VISUAL_RGB
)

# Get order status
order_status = client.ordering.edc.get(order.id)

# Get detailed line items
line_items = client.ordering.edc.get_line_items(order.id)
for item in line_items:
    print(f"{item.order_name}: {item.state}")
```

### Using STAC Items from Search

```python
from earthdaily import EDSClient, EDSConfig

config = EDSConfig()
client = EDSClient(config)

# Search for items
search = client.platform.pystac_client.search(
    collections=["catalog-collection"],
    datetime="2024-01-01/2024-01-31",
    max_items=10
)

items = list(search.items())

# Create order from search results
order = client.ordering.edc.create(
    items=items,
    product_type="VISUAL_RGB"
)
```

