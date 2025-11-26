# Migration Guide: EarthDaily Python Client v0 to v1

[![PyPI version](https://badge.fury.io/py/earthdaily.svg)](https://badge.fury.io/py/earthdaily)
[![Documentation](https://img.shields.io/badge/Documentation-Online-brightgreen.svg)](https://earthdaily.github.io/earthdaily-python-client/)
![Python Versions](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)

Welcome to the EarthDaily Python Client v1! This guide will help you migrate your existing v0 code to take advantage of the new features and improvements in v1.

## 🚀 What's New in v1

v1 represents a major evolution of the EarthDaily Python client with:

- **Platform API Integration**: Direct access to the EarthDaily platform APIs
- **STAC Item Management**: Full CRUD operations for STAC items
- **Improved Architecture**: Cleaner, more maintainable codebase
- **Enhanced Error Handling**: Better exception handling and error messages
- **Modern Configuration**: Streamlined configuration management

## 📦 Installation

**Python Version Support**: Python 3.10, 3.11, 3.12, 3.13, 3.14

### Basic Installation

```bash
pip install earthdaily
```

### Installation with Optional Dependencies

For **platform API functionality** (recommended for v1 features):
```bash
pip install "earthdaily[platform]"
```

For **legacy v0 functionality** (if you need backward compatibility):
```bash
pip install "earthdaily[legacy]"
```

For **both platform and legacy features**:
```bash
pip install "earthdaily[platform,legacy]"
```

For **full features with utils (for .env file and Jupyter notebooks)**:
```bash
pip install "earthdaily[platform,legacy,utils]"
```

## 🔄 Key Differences

### v0 vs v1 Architecture

| Aspect | v0 (Legacy) | v1 (Current) |
|--------|-------------|--------------|
| **Main Entry Point** | Various modules | `EDSClient` |
| **Configuration** | Manual setup | `EDSConfig` |
| **API Access** | Datacube functionality | Full platform API access |
| **STAC Operations** | Read-only via STAC client | Full CRUD operations |
| **Error Handling** | Basic exceptions | Comprehensive with `EDSAPIError` |
| **Installation** | Single package | Optional dependencies (`platform`, `legacy`) |

## 🛠️ Migration Steps

### Step 1: Update Your Imports

**v0:**
```python
# Old v0 imports
import earthdaily
from earthdaily import EarthDataStore, datasets
```

**v1:**
```python
# New v1 imports
from earthdaily import EDSClient, EDSConfig
from earthdaily.exceptions import EDSAPIError
```

### Step 2: Initialize the Client

**v0:**
```python
# Old way - direct initialization
eds = EarthDataStore()
```

**v1:**
```python
# New way - unified client initialization
config = EDSConfig()
client = EDSClient(config)
```

### Step 3: Authentication

Both v0 and v1 support environment variable authentication. **You need a `.env` file**:

```bash
# .env file (required)
EDS_CLIENT_ID=your_client_id
EDS_SECRET=your_client_secret
EDS_AUTH_URL=https://your-auth-url.com/oauth/token
EDS_API_URL=https://api.earthdaily.com
```

Load environment variables in your code:
```python
from dotenv import load_dotenv
load_dotenv(".env")  # Required for authentication
```

### Step 4: Update Your Data Access Patterns

#### Creating Datacubes

**v0:**
```python
# Load geometry and initialize
geometry = datasets.load_pivot()
eds = EarthDataStore()

# Create datacube in one step
s2_datacube = eds.datacube(
    "sentinel-2-l2a",
    assets=["blue", "green", "red", "nir"],
    intersects=geometry,
    datetime=["2022-08-01", "2022-08-09"],
    mask_with="native",
    clear_cover=50,
)
```

**v1:**
```python
# Load geometry and initialize client
from earthdaily.legacy.datasets import load_pivot

geometry = load_pivot()
client = EDSClient(EDSConfig())

# Create datacube using legacy functionality in v1
s2_datacube = client.legacy.datacube(
    "sentinel-2-l2a",
    assets=["blue", "green", "red", "nir"],
    intersects=geometry,
    datetime=["2022-08-01", "2022-08-09"],
    mask_with="native",
    clear_cover=50,
)
```

#### Searching for Items

**v0:**
```python
items = eds.search(
    "sentinel-2-l2a",
    intersects=geometry,
    datetime=["2022-08-01", "2022-08-09"],
)
```

**v1 (Platform API):**
```python
search_result = client.platform.pystac_client.search(
    collections=["sentinel-2-l2a"],
    datetime="2022-08-01T00:00:00Z/2022-08-09T00:00:00Z",
    max_items=50,
)
items = list(search_result.items())
```

**v1 (Legacy):**
```python
items = client.legacy.search(
    "sentinel-2-l2a",
    intersects=geometry,
    datetime=["2022-08-01", "2022-08-09"],
)
```

## 📋 v1 Examples

**Note**: To use `.env` files and Jupyter notebooks, install with the `utils` extra:
```bash
pip install "earthdaily[utils]"
```

### Basic Client Setup
```python
from dotenv import load_dotenv
from earthdaily import EDSClient, EDSConfig

load_dotenv(".env")  # Load credentials
client = EDSClient(EDSConfig())
```

### STAC Item Management (New in v1)
```python
# Create a STAC item
stac_item = {
    "type": "Feature",
    "stac_version": "1.0.0",
    "id": "example-item-123",
    "collection": "eda-labels-vessels",
    "geometry": {"type": "Point", "coordinates": [-67.7, -37.8]},
    "properties": {"datetime": "2017-12-08T14:38:16.000000Z"},
    "links": [],
    "assets": {},
}

client.platform.stac_item.create_item("eda-labels-vessels", stac_item)
```

### Search with Platform API
```python
search_result = client.platform.pystac_client.search(
    collections=["sentinel-2-l2a"],
    datetime="2024-06-01T00:00:00Z/2024-08-01T00:00:00Z",
    max_items=10,
)
items = list(search_result.items())
```

### Legacy Datacube (v0 functionality in v1)
```python
from earthdaily.legacy.datasets import load_pivot

geometry = load_pivot()
datacube = client.legacy.datacube(
    "sentinel-2-l2a",
    assets=["blue", "green", "red"],
    intersects=geometry,
    datetime=["2022-08-01", "2022-08-09"],
    mask_with="native",
)
```

## 🔧 Legacy Support

Your v0 functionality is now accessible via `client.legacy` in v1:

```python
# v0 code
eds = EarthDataStore()
items = eds.search("sentinel-2-l2a", intersects=geometry)

# v1 equivalent - same functionality via client.legacy
client = EDSClient(EDSConfig())
items = client.legacy.search("sentinel-2-l2a", intersects=geometry)
```

All v0 methods are available through `client.legacy.*`

## ⚠️ Breaking Changes

1. **Initialization**: `EarthDataStore()` → `EDSClient(EDSConfig())`
2. **Access Pattern**: `eds.method()` → `client.legacy.method()` for v0 functionality
3. **Configuration**: Environment variables now required (`.env` file)
4. **Installation**: Optional dependencies `[platform]` and `[legacy]` available

## 📖 Additional Resources

- [API Documentation](https://earthdaily.github.io/earthdaily-python-client/)
- [Quick Start Example](https://github.com/earthdaily/earthdaily-python-client/blob/main/examples/quick_start.py)
- [Contributing Guide](https://github.com/earthdaily/earthdaily-python-client/blob/main/CONTRIBUTING.md)
- [GitHub Issues](https://github.com/earthdaily/earthdaily-python-client/issues)

## 🆘 Need Help?

If you encounter issues during migration:

1. Check the [GitHub Issues](https://github.com/earthdaily/earthdaily-python-client/issues)
2. Review the [examples](https://github.com/earthdaily/earthdaily-python-client/tree/main/examples/) directory
3. Create a new issue with your specific migration question

Welcome to EarthDaily Python Client v1! 🎉 