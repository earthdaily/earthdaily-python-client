# EarthDaily Python Client

[![PyPI version](https://badge.fury.io/py/earthdaily.svg)](https://badge.fury.io/py/earthdaily)
[![Documentation](https://img.shields.io/badge/Documentation-Online-brightgreen.svg)](https://earthdaily.github.io/earthdaily-python-client/)
![Python Versions](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)

The EarthDaily Python Client is a comprehensive library for interacting with the EarthDaily Analytics platform. It provides seamless access to satellite data, STAC item management, and platform APIs through a unified interface.

## 🚀 Key Features

- **Platform API Access**: Full integration with EarthDaily platform services
- **STAC Item Management**: Complete CRUD operations for STAC items
- **Datacube Module**: Modern, self-contained datacube creation and manipulation
- **Modern Architecture**: Streamlined client design with comprehensive error handling
- **Flexible Installation**: Modular installation options for different use cases

## 📦 Installation

**Supported Python Versions**: 3.10, 3.11, 3.12, 3.13, 3.14

### Basic Installation
```bash
pip install earthdaily
```

### Recommended Installation (Platform Features)
```bash
pip install "earthdaily[platform]"
```

### Datacube Support
```bash
pip install "earthdaily[datacube]"
```

### Full Installation (All Features)
```bash
pip install "earthdaily[platform,datacube]"
```

### Full Installation with utils (for .env file and Jupyter notebooks)
```bash
pip install "earthdaily[platform,datacube,utils]"
```

> **Looking for v1/legacy functionality?** The legacy module (`client.legacy`) was removed in v2.0.0.
> If you need legacy datacube features, pin to v1.x.x (`earthdaily<2.0.0`).
> The v1.x.x line will continue to receive security and critical bug fixes only.

## Environment Setup

The EarthPlatform STAC API is protected by bearer authentication. A bearer token is generated using OAuth Client Credentials Flow. The required `client_id`, `client_secret`, and `access_token_url` values can be found on the [Account Management Console](https://console.earthdaily.com/account).

### Getting Your Credentials

1. Go to the [Account Management Console](https://console.earthdaily.com/account)
2. Click the **Provision New API Credentials** button to generate your API credentials
3. Copy the generated `client_secret` value — this will be your `EDS_SECRET`

For more details, see the [API Authentication Guide](https://earthdaily.github.io/EDA-Documentation/GettingStarted/APIAuthentication/).

### Configure Your Environment

Create a `.env` file in your project root with your credentials:

```bash
# .env
EDS_CLIENT_ID=EARTHDAILY_API_TOKEN
EDS_SECRET=<Go to Account Management Console, click Provision New API Credentials, copy the client_secret>
EDS_AUTH_URL=https://api.earthdaily.com/account_management/v1/authentication/api_tokens/exchange
EDS_API_URL=https://api.earthdaily.com
```

**Note**: To use `.env` files and Jupyter notebooks, install with the `utils` extra:
```bash
pip install "earthdaily[utils]"
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
    client_id="EARTHDAILY_API_TOKEN",
    client_secret="<Go to Account Management Console, click Provision New API Credentials, copy the client_secret>",
    token_url="https://api.earthdaily.com/account_management/v1/authentication/api_tokens/exchange",
    base_url="https://api.earthdaily.com"
)
client = EDSClient(config)
```

### Advanced Configuration Options

The `EDSConfig` class supports additional configuration options for customizing client behavior:

```python
config = EDSConfig(
    # ... authentication parameters ...
    
    # HTTP retry configuration
    max_retries=5,                    # Maximum retry attempts (default: 3)
    retry_backoff_factor=2.0,         # Exponential backoff factor (default: 1.0)
    
    # Asset access mode
    asset_access_mode="presigned-urls"  # "presigned-urls", "proxy-urls", or "raw"
)
```

#### Retry Configuration

- **`max_retries`**: Maximum number of retry attempts
- **`retry_backoff_factor`**: Backoff factor for retry delays

**Examples:**
- `retry_backoff_factor=1.0`: Delays of 1s, 2s, 4s
- `retry_backoff_factor=0.5`: Delays of 0.5s, 1s, 2s  
- `retry_backoff_factor=2.0`: Delays of 2s, 4s, 8s

### 🍪 Proxy Asset URLs And Cookie File

When `asset_access_mode="proxy-urls"`, assets are served through authenticated
proxy URLs. GDAL/rasterio cannot send the EDS bearer token, so the client can
fetch a short-lived asset cookie and write it to a Netscape-format file that
you pass through `GDAL_HTTP_COOKIEFILE`.

Enable the feature on `EDSConfig` and ask the client for a cookie file. You
must provide `path` explicitly so the credential-like file does not end up in
an unexpected location (for example, committed to a repository):

```python
import rasterio
from earthdaily import EDSClient, EDSConfig

config = EDSConfig(
    asset_access_mode="proxy-urls",
    platform_cookie_enabled=True,
    platform_cookie_auto_refresh=True,
)

with EDSClient(config) as client:
    cookie_file = client.get_platform_cookie_file(
        path="/tmp/earthdaily-platform-cookies.txt"
    )

    with rasterio.Env(GDAL_HTTP_COOKIEFILE=str(cookie_file)):
        ...  # e.g. odc.stac.stac_load(...), rasterio.open(...)
```

- `platform_cookie_auto_refresh=True` starts a background daemon thread that
  refreshes the cookie file in place before it expires.
- `EDSClient.close()` (called automatically by the `with` statement) stops the
  refresh thread and deletes the cookie file.
- The parent directory of `path` must already exist; the client never creates
  directories for you.

See the [EarthMosaics notebook](examples/applications/Introduction%20to%20EDA%20EarthMosaics/Introduction%20to%20EDA%20EarthMosaics.ipynb)
for a full example.

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

## 🏗️ Architecture Overview

The client is organized into main modules:

- **`client.platform`**: Modern platform API access
  - `pystac_client`: STAC catalog search
  - `stac_item`: STAC item CRUD operations
  - `bulk_search`: Bulk search operations
  - `bulk_insert`: Bulk data insertion
  - `bulk_delete`: Bulk data deletion

- **`client.datacube`**: Datacube creation and manipulation
  - `create()`: Create analysis-ready datacubes from STAC items
  - Masking, temporal aggregation, spectral indices, zonal statistics, and more

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
    export_format="stacjson"
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
    client.platform.stac_item.download_assets(
        item,
        asset_keys=["blue", "green", "red"],
        output_dir="./downloads",
        max_workers=3
    )
```

## 📚 Documentation & Examples

- [📖 **EDA Documentation**](https://earthdaily.github.io/EDA-Documentation/)
- [📖 **Full API Documentation**](https://earthdaily.github.io/earthdaily-python-client/)
- [📝 **Quick Start Examples**](https://github.com/earthdaily/earthdaily-python-client/tree/main/examples/)
- [🧪 **Jupyter Notebooks**](https://github.com/earthdaily/earthdaily-python-client/tree/main/examples/applications/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/earthdaily/earthdaily-python-client/blob/main/CONTRIBUTING.md) for details on:

- Development setup
- Code style guidelines
- Testing procedures
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/earthdaily/earthdaily-python-client/blob/main/LICENSE) file for details.

## 🆘 Support

Need help? Here's how to get support:

- 📖 Check the [documentation](https://earthdaily.github.io/earthdaily-python-client/)
- 🐛 [Open an issue](https://github.com/earthdaily/earthdaily-python-client/issues/new) for bugs
- 💬 Ask questions in [GitHub Discussions](https://github.com/earthdaily/earthdaily-python-client/discussions)

---

**Ready to get started?** Check out our [Quick Start Example](https://github.com/earthdaily/earthdaily-python-client/blob/main/examples/quick_start.py) or explore the [API Documentation](https://earthdaily.github.io/earthdaily-python-client/)! 🚀
