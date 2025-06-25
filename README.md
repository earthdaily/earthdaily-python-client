# EarthDaily Python Client

[![PyPI version](https://badge.fury.io/py/earthdaily.svg)](https://badge.fury.io/py/earthdaily)
[![Documentation](https://img.shields.io/badge/Documentation-Online-brightgreen.svg)](https://earthdaily.github.io/earthdaily-python-client/)
![Python Versions](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)

The EarthDaily Python Client is a comprehensive library for interacting with the EarthDaily Analytics platform. It provides seamless access to satellite data, STAC item management, and platform APIs through a unified interface.

## 🚀 Key Features

- **Platform API Access**: Full integration with EarthDaily platform services
- **STAC Item Management**: Complete CRUD operations for STAC items
- **Legacy Support**: Backward compatibility with v0 datacube functionality
- **Modern Architecture**: Streamlined client design with comprehensive error handling
- **Flexible Installation**: Modular installation options for different use cases

## 📦 Installation

**Supported Python Versions**: 3.10, 3.11, 3.12, 3.13

### Basic Installation
```bash
pip install earthdaily
```

### Recommended Installation (Platform Features)
```bash
pip install earthdaily[platform]
```

### Legacy Support (v0 Compatibility)
```bash
pip install earthdaily[legacy]
```

### Full Installation (All Features)
```bash
pip install earthdaily[platform,legacy]
```

## 🔧 Environment Setup

Create a `.env` file in your project root with your credentials:

```bash
# .env
EDS_CLIENT_ID=your_client_id
EDS_SECRET=your_client_secret
EDS_AUTH_URL=https://your-auth-url.com/oauth/token
EDS_API_URL=https://api.earthdaily.com
```

## 🏃 Quick Start

```python
from dotenv import load_dotenv
from earthdaily import EDSClient, EDSConfig

# Load environment variables
load_dotenv(".env")

# Initialize client
config = EDSConfig()
client = EDSClient(config)
```

### Alternative Configuration
```python
# Direct configuration (without .env file)
config = EDSConfig(
    client_id="your_client_id",
    client_secret="your_client_secret",
    token_url="https://your-auth-url.com/oauth/token",
    api_url="https://api.earthdaily.com"
)
client = EDSClient(config)
```

## 🌍 Core Features

### Platform API Integration

Search for satellite data using STAC:
```python
# Search for Sentinel-2 data
search_result = client.platform.pystac_client.search(
    collections=["sentinel-2-l2a"],
    datetime="2024-06-01T00:00:00Z/2024-08-01T00:00:00Z",
    max_items=10
)
items = list(search_result.items())
```

### STAC Item Management

Create and manage STAC items:
```python
# Create a new STAC item
stac_item = {
    "type": "Feature",
    "stac_version": "1.0.0",
    "id": "example-item-123",
    "collection": "your-collection",
    "geometry": {"type": "Point", "coordinates": [-67.7, -37.8]},
    "properties": {"datetime": "2024-01-01T00:00:00Z"},
    "links": [],
    "assets": {}
}

client.platform.stac_item.create_item("your-collection", stac_item)
```

### Legacy Datacube Support

Access v0 functionality through the legacy interface:
```python
from earthdaily.legacy.datasets import load_pivot

# Load geometry and create datacube
geometry = load_pivot()
datacube = client.legacy.datacube(
    "sentinel-2-l2a",
    assets=["blue", "green", "red", "nir"],
    intersects=geometry,
    datetime=["2022-08-01", "2022-08-09"],
    mask_with="native"
)
```

## 🏗️ Architecture Overview

The client is organized into main modules:

- **`client.platform`**: Modern platform API access
  - `pystac_client`: STAC catalog search
  - `stac_item`: STAC item CRUD operations
  - `bulk_search`: Bulk search operations
  - `bulk_insert`: Bulk data insertion
  - `bulk_delete`: Bulk data deletion

- **`client.legacy`**: v0 compatibility layer
  - `datacube()`: Create analysis-ready datacubes
  - `search()`: Legacy search functionality
  - Access to existing v0 methods

## 🔧 Platform API Methods

### STAC Item Management (`client.platform.stac_item`)

#### Create Items
```python
# Create a new STAC item
item = client.platform.stac_item.create_item(
    collection_id="your-collection",
    item_data={
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "item-123",
        "geometry": {"type": "Point", "coordinates": [-67.7, -37.8]},
        "properties": {"datetime": "2024-01-01T00:00:00Z"}
    },
    return_format="dict"  # "dict", "json", or "pystac"
)
```

#### Read Items
```python
# Get a specific item
item = client.platform.stac_item.get_item(
    collection_id="your-collection",
    item_id="item-123",
    return_format="pystac"
)
```

#### Update Items
```python
# Update an existing item
updated_item = client.platform.stac_item.update_item(
    collection_id="your-collection",
    item_id="item-123",
    item_data={"properties": {"updated": "2024-01-02T00:00:00Z"}},
    return_format="dict"
)
```

#### Delete Items
```python
# Delete an item
client.platform.stac_item.delete_item(
    collection_id="your-collection",
    item_id="item-123"
)
```

#### Download Assets
```python
# Download item assets
downloads = client.platform.stac_item.download_assets(
    item=item,
    asset_keys=["blue", "green", "red"],
    output_dir="./downloads",
    max_workers=3
)
```

### Bulk Search (`client.platform.bulk_search`)

#### Create Bulk Search
```python
# Create a bulk search job
search_job = client.platform.bulk_search.create(
    collections=["sentinel-2-l2a"],
    datetime="2024-01-01T00:00:00Z/2024-02-01T00:00:00Z",
    bbox=[-74.2, 40.6, -73.9, 40.9],  # NYC area
    limit=1000,
    export_format="json"
)
print(f"Job ID: {search_job.job_id}")
```

#### Monitor Job Status
```python
# Check job status
job_status = client.platform.bulk_search.fetch(search_job.job_id)
print(f"Status: {job_status.status}")
print(f"Assets: {len(job_status.assets)}")
```

#### Download Results
```python
# Download search results when completed
if job_status.status == "COMPLETED":
    job_status.download_assets(save_location=Path("./bulk_results"))
```

### Bulk Insert (`client.platform.bulk_insert`)

#### Create Bulk Insert Job
```python
# Create bulk insert job
insert_job = client.platform.bulk_insert.create(
    collection_id="your-collection",
    error_handling_mode="CONTINUE",  # or "STOP"
    conflict_resolution_mode="SKIP"  # or "OVERRIDE"
)
```

#### Upload Data
```python
# Prepare STAC items file and upload
items_file = Path("./stac_items.jsonl")  # JSONL format
insert_job.upload(items_file)

# Start the job
insert_job.start()
```

#### Monitor Insert Progress
```python
# Check insert job status
job_status = client.platform.bulk_insert.fetch(insert_job.job_id)
print(f"Items written: {job_status.items_written_count}")
print(f"Errors: {job_status.items_error_count}")
```

### Bulk Delete (`client.platform.bulk_delete`)

#### Create Bulk Delete Job
```python
# Create bulk delete job
delete_job = client.platform.bulk_delete.create(
    collection_id="your-collection"
)
```

#### Upload Item IDs
```python
# Prepare file with item IDs to delete
ids_file = Path("./items_to_delete.txt")
delete_job.upload(ids_file)

# Start the deletion
delete_job.start()
```

#### Monitor Deletion Progress
```python
# Check delete job status
job_status = client.platform.bulk_delete.fetch(delete_job.job_id)
print(f"Items deleted: {job_status.items_deleted_count}")
print(f"Errors: {job_status.items_error_count}")
```

### STAC Catalog Search (`client.platform.pystac_client`)

#### Standard STAC Search
```python
# Search for items using STAC API
search_results = client.platform.pystac_client.search(
    collections=["sentinel-2-l2a"],
    datetime="2024-01-01T00:00:00Z/2024-02-01T00:00:00Z",
    bbox=[-74.2, 40.6, -73.9, 40.9],
    max_items=50
)

# Process results
items = list(search_results.items())
print(f"Found {len(items)} items")
```

#### Get Collections
```python
# List available collections
collections = client.platform.pystac_client.get_collections()
for collection in collections:
    print(f"Collection: {collection.id}")
```

## 🔄 Legacy Methods (`client.legacy`)

### Create Datacubes
```python
from earthdaily.legacy.datasets import load_pivot

# Load sample geometry
geometry = load_pivot()

# Create analysis-ready datacube
datacube = client.legacy.datacube(
    collections="sentinel-2-l2a",
    assets=["blue", "green", "red", "nir"],
    intersects=geometry,
    datetime=["2022-08-01", "2022-08-09"],
    mask_with="native",  # Apply cloud masking
    clear_cover=50,      # Minimum 50% clear pixels
    groupby_date="mean"  # Aggregate by date
)
```

### Search Items
```python
# Search for items (legacy interface)
items = client.legacy.search(
    collections="sentinel-2-l2a",
    intersects=geometry,
    datetime=["2022-08-01", "2022-08-09"],
    limit=100
)
print(f"Found {len(items)} items")
```

### Multi-Collection Datacubes
```python
# Create datacube from multiple collections
datacube = client.legacy.datacube(
    collections=["sentinel-2-l2a", "landsat-c2l2-sr"],
    assets=["red", "green", "blue"],
    intersects=geometry,
    datetime="2022-08",
    cross_calibration_collection="landsat-c2l2-sr"
)
```

## 🔍 Usage Examples

### Data Discovery
```python
# Find available collections
collections = client.platform.pystac_client.get_collections()
print([c.id for c in collections])
```

### Download Data
```python
# Download assets from search results
for item in items:
    client.platform.stac_item.download_item_assets(
        item,
        assets=["blue", "green", "red"],
        path="./downloads"
    )
```

### Error Handling
```python
from earthdaily.exceptions import EDSAPIError

try:
    search_result = client.platform.pystac_client.search(
        collections=["invalid-collection"]
    )
except EDSAPIError as e:
    print(f"API Error: {e}")
```

## 📚 Documentation & Examples

- [📖 **Full API Documentation**](https://earthdaily.github.io/earthdaily-python-client/)
- [🔄 **Migration Guide (v0 → v1)**](docs/migration-v0-to-v1.md)
- [📝 **Quick Start Examples**](examples/)
- [🧪 **Jupyter Notebooks**](custom_examples/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development setup
- Code style guidelines
- Testing procedures
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

Need help? Here's how to get support:

- 📖 Check the [documentation](https://earthdaily.github.io/earthdaily-python-client/)
- 🐛 [Open an issue](https://github.com/earthdaily/earthdaily-python-client/issues/new) for bugs
- 💬 Ask questions in [GitHub Discussions](https://github.com/earthdaily/earthdaily-python-client/discussions)
- 📧 Contact our support team

---

**Ready to get started?** Check out our [Quick Start Example](examples/quick_start.py) or explore the [API Documentation](https://earthdaily.github.io/earthdaily-python-client/)! 🚀
