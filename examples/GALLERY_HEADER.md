# EarthDaily Python Client Examples

Welcome to the EarthDaily Python Client examples! These examples demonstrate the key features and capabilities of the v1 client.

## Available Examples

### 🚀 [Quick Start](quick_start.py)
Get started quickly with searching and exploring STAC items using the new v1 API.

### 📊 [Datacube Creation](datacube_example.py) 
Learn how to create datacubes using the legacy API for backward compatibility with v0.x code.

### 📥 [Asset Download](asset_download_example.py)
Download satellite data assets with different access modes and configurations.

### 🔍 [Bulk Search Operations](bulk_search_example.py)
Perform large-scale searches and downloads using the bulk search functionality.

## Getting Started

1. **Set up your credentials** as environment variables:
   ```bash
   export EDS_CLIENT_ID="your_client_id"
   export EDS_SECRET="your_client_secret"  
   export EDS_AUTH_URL="your_auth_url"
   export EDS_API_URL="https://api.earthdaily.com"
   ```

2. **Install the required dependencies**:
   ```bash
   # For all examples
   pip install "earthdaily[platform,legacy]"
   ```

3. **Run an example**:
   ```bash
   python quick_start.py
   ```

Each example includes interactive demos and detailed explanations to help you understand the EarthDaily Python client capabilities.