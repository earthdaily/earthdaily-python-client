"""
Datacube Creation Example
=========================

This example demonstrates how to create datacubes using the EarthDaily Python client v1
with the legacy API for backward compatibility.

Features demonstrated:
- Using the legacy API for v0.x compatibility
- Loading sample geometries
- Creating datacubes with various parameters
- Working with masking and cloud filtering
- Basic datacube operations

Requirements:
- Set your EDS credentials as environment variables or in a .env file
- Install with legacy support: pip install 'earthdaily[legacy]'
"""

import warnings

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("💡 Consider installing the utils extra to automatically load .env files:")
    print("   pip install 'earthdaily[legacy,utils]'")

from earthdaily import EDSClient, EDSConfig
from earthdaily.exceptions import EDSAPIError
from earthdaily.legacy import datasets

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def initialize_client():
    """Initialize the EarthDaily API client."""
    print("🚀 Initializing EarthDaily Client with legacy support...")
    config = EDSConfig()
    client = EDSClient(config)
    print("✅ Client initialized successfully!")
    return client


def load_sample_geometry():
    """Load a sample geometry for testing."""
    print("\n📍 Loading sample pivot geometry...")

    # Load the built-in pivot geometry
    pivot = datasets.load_pivot()
    try:
        # Try to access geometry attributes
        geom_type = getattr(pivot, "geometry", None)
        bounds = getattr(pivot, "total_bounds", None)
        if geom_type is not None and bounds is not None:
            print(f"   Geometry type: {geom_type.iloc[0].geom_type}")
            print(f"   Bounds: {bounds}")
        else:
            print("   Sample geometry loaded")
    except (AttributeError, IndexError):
        print("   Sample geometry loaded")

    return pivot


def create_basic_datacube(client, geometry):
    """Create a basic datacube with Sentinel-2 data."""
    print("\n🔍 Creating basic Sentinel-2 datacube...")

    try:
        datacube = client.legacy.datacube(
            collections="sentinel-2-l2a",
            intersects=geometry,
            datetime=["2023-06-01", "2023-08-01"],  # Summer months
            assets=["red", "green", "blue", "nir"],
            rescale=True,  # Convert to reflectance values
        )

        print("✅ Datacube created successfully!")
        print(f"   Shape: {datacube.sizes}")
        print(f"   Variables: {list(datacube.data_vars)}")
        print(f"   Time range: {datacube.time.min().values} to {datacube.time.max().values}")
        print(f"   Coordinate system: {datacube.rio.crs}")
        print(f"   Spatial resolution: {abs(datacube.rio.resolution()[0]):.1f}m")

        return datacube

    except Exception as e:
        print(f"❌ Error creating basic datacube: {e}")
        return None


def create_masked_datacube(client, geometry):
    """Create a datacube with cloud masking."""
    print("\n☁️ Creating cloud-masked datacube...")

    try:
        datacube = client.legacy.datacube(
            collections="sentinel-2-l2a",
            intersects=geometry,
            datetime=["2023-06-01", "2023-08-01"],
            assets=["red", "green", "blue", "nir"],
            mask_with="native",  # Use native SCL mask
            clear_cover=30,  # Require at least 30% clear pixels
            rescale=True,
        )

        print("✅ Masked datacube created successfully!")
        print(f"   Shape: {datacube.sizes}")
        print(f"   Variables: {list(datacube.data_vars)}")
        print(f"   Clear pixel percentages: {datacube.clear_percent.values}")

        return datacube

    except Exception as e:
        print(f"❌ Error creating masked datacube: {e}")
        return None


def analyze_datacube(datacube, name="datacube"):
    """Perform basic analysis on a datacube."""
    if datacube is None:
        return

    print(f"\n📊 Analyzing {name}...")

    # Calculate NDVI
    if "red" in datacube.data_vars and "nir" in datacube.data_vars:
        ndvi = (datacube.nir - datacube.red) / (datacube.nir + datacube.red)
        print(f"   NDVI range: {float(ndvi.min()):.3f} to {float(ndvi.max()):.3f}")
        print(f"   Mean NDVI: {float(ndvi.mean()):.3f}")

    # Show temporal information
    if len(datacube.time) > 1:
        print(f"   Number of time steps: {len(datacube.time)}")
        # Calculate time resolution in days
        time_diff = datacube.time[1] - datacube.time[0]
        days = time_diff.values.astype("timedelta64[D]").astype(int)
        print(f"   Time resolution: ~{days} days")

    # Show spatial information
    print(f"   Spatial dimensions: {datacube.sizes['x']} x {datacube.sizes['y']} pixels")
    print(f"   Total pixels per time step: {datacube.sizes['x'] * datacube.sizes['y']:,}")


def demonstrate_search_functionality(client, geometry):
    """Demonstrate the search functionality without creating a datacube."""
    print("\n🔍 Demonstrating search functionality...")

    try:
        # Search for items
        items = client.legacy.search(
            collections="sentinel-2-l2a",
            intersects=geometry,
            datetime=["2023-07-01", "2023-07-31"],
            query={"eo:cloud_cover": {"lt": 20}},  # Low cloud cover
        )

        print(f"✅ Found {len(items)} items with <20% cloud cover")

        # Show details of first few items
        for i, item in enumerate(items[:3]):
            print(f"   {i + 1}. {item.id}")
            print(f"      Date: {item.datetime}")
            print(f"      Cloud cover: {item.properties.get('eo:cloud_cover', 'N/A')}%")

        return items

    except Exception as e:
        print(f"❌ Error in search: {e}")
        return []


def main():
    """Main function to demonstrate datacube creation workflows."""
    try:
        # Initialize client
        client = initialize_client()

        # Load sample geometry
        geometry = load_sample_geometry()

        # Demonstrate search functionality
        items = demonstrate_search_functionality(client, geometry)

        if len(items) > 0:
            print(f"\n✅ Found {len(items)} suitable items. Proceeding with datacube creation...")

            # Create basic datacube
            basic_cube = create_basic_datacube(client, geometry)
            analyze_datacube(basic_cube, "basic datacube")

            # Create masked datacube
            masked_cube = create_masked_datacube(client, geometry)
            analyze_datacube(masked_cube, "masked datacube")

            print("\n✨ Datacube example completed successfully!")
            print("💡 Next steps:")
            print("   - Try different collections (sentinel-1-rtc, landsat-c2l2-sr)")
            print("   - Experiment with different time periods")
            print("   - Use your own geometries")
            print("   - Calculate vegetation indices or other analytics")

        else:
            print("\n❌ No suitable items found for the specified criteria.")
            print("💡 Try adjusting the date range or cloud cover threshold.")

    except EDSAPIError as e:
        print(f"\n❌ API Error: {e}")
        print(f"   Status Code: {e.status_code}")
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("💡 Make sure to install with legacy support:")
        print("   pip install 'earthdaily[legacy]'")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("\n💡 Make sure you have set your EDS credentials as environment variables:")
        print("   EDS_CLIENT_ID, EDS_SECRET, EDS_AUTH_URL, EDS_API_URL")


if __name__ == "__main__":
    main()
