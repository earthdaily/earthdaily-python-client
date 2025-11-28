"""
Bulk Search Example
===================

This example demonstrates how to use the EarthDaily Python client v1
to perform bulk search operations for large-scale data discovery and download.

Features demonstrated:
- Creating bulk search jobs
- Monitoring job status
- Downloading results
- Error handling for bulk operations

Requirements:
- Set your EDS credentials as environment variables or in a .env file
- Install with platform support: pip install 'earthdaily[platform]'
"""

import time
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("💡 Consider installing the utils extra to automatically load .env files:")
    print("   pip install 'earthdaily[platform,utils]'")

from earthdaily import EDSClient, EDSConfig
from earthdaily.exceptions import EDSAPIError


def initialize_client():
    """Initialize the EarthDaily API client."""
    print("🚀 Initializing EarthDaily Client for bulk operations...")
    config = EDSConfig()
    client = EDSClient(config)
    print("✅ Client initialized successfully!")
    return client


def create_bulk_search_job(client, collections, datetime_range=None, limit=100):
    """Create a bulk search job for the specified parameters."""
    print(f"\n📋 Creating bulk search job for {collections}...")

    try:
        # Create bulk search job
        bulk_search_response = client.platform.bulk_search.create(
            collections=collections, export_format="stacjson", datetime=datetime_range, limit=limit
        )

        job_id = bulk_search_response.job_id
        print("✅ Bulk search job created successfully!")
        print(f"   Job ID: {job_id}")
        print(f"   Collection(s): {collections}")
        print("   Export format: stacjson")
        print(f"   Limit: {limit} items")
        if datetime_range:
            print(f"   Date range: {datetime_range}")

        return job_id

    except EDSAPIError as e:
        print(f"❌ Error creating bulk search job: {e}")
        print(f"   Status Code: {e.status_code}")
        print(f"   Details: {e.body}")
        return None
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return None


def monitor_job_status(client, job_id, max_wait_time=300, check_interval=5):
    """Monitor the status of a bulk search job."""
    print(f"\n⏳ Monitoring job status (Job ID: {job_id})...")

    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        try:
            # Fetch job status
            bulk_search_job = client.platform.bulk_search.fetch(job_id)
            status = bulk_search_job.status

            print(f"   Status: {status}")

            if status == "COMPLETED":
                print("✅ Job completed successfully!")
                return bulk_search_job
            elif status == "FAILED":
                print("❌ Job failed!")
                if hasattr(bulk_search_job, "error_message"):
                    print(f"   Error: {bulk_search_job.error_message}")
                return None
            elif status in ["PENDING", "IN_PROGRESS"]:
                print(f"   Job is {status.lower()}... waiting {check_interval} seconds")
                time.sleep(check_interval)
            else:
                print(f"   Unknown status: {status}")
                time.sleep(check_interval)

        except EDSAPIError as e:
            print(f"❌ Error fetching job status: {e}")
            return None
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            return None

    print(f"⏰ Job did not complete within {max_wait_time} seconds")
    return None


def download_job_results(bulk_search_job, output_directory):
    """Download the results of a completed bulk search job."""
    print("\n📥 Downloading job results...")

    try:
        # Ensure output directory exists
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        # Download assets
        bulk_search_job.download_assets(save_location=output_path)

        print("✅ Results downloaded successfully!")
        print(f"   Output directory: {output_path.absolute()}")

        # List downloaded files
        downloaded_files = list(output_path.glob("*"))
        if downloaded_files:
            print("   Downloaded files:")
            for file in downloaded_files[:5]:  # Show first 5 files
                print(f"     - {file.name}")
            if len(downloaded_files) > 5:
                print(f"     ... and {len(downloaded_files) - 5} more files")

        return True

    except Exception as e:
        print(f"❌ Error downloading results: {e}")
        return False


def demo_small_bulk_search():
    """Demonstrate bulk search with a small dataset."""
    print("=" * 60)
    print("🔍 DEMO: Small Bulk Search (GOES Collection)")
    print("=" * 60)

    client = initialize_client()

    # Create job for GOES collection with limit
    job_id = create_bulk_search_job(
        client=client,
        collections=["goes"],
        limit=10,  # Small limit for demo
        datetime_range="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
    )

    if job_id:
        # Monitor job
        completed_job = monitor_job_status(client, job_id, max_wait_time=120)

        if completed_job:
            # Download results to a demo directory
            output_dir = Path.home() / "Downloads" / "earthdaily_bulk_search_demo"
            success = download_job_results(completed_job, output_dir)

            if success:
                print("\n✨ Demo completed successfully!")
                print(f"💡 Check the downloaded files in: {output_dir}")
            else:
                print("\n❌ Demo completed but download failed")
        else:
            print("\n❌ Demo failed - job did not complete successfully")
    else:
        print("\n❌ Demo failed - could not create job")


def demo_custom_bulk_search():
    """Demonstrate bulk search with custom parameters."""
    print("\n" + "=" * 60)
    print("🛠️  DEMO: Custom Bulk Search Configuration")
    print("=" * 60)

    # Get user input for demonstration
    print("\nThis demo allows you to configure bulk search parameters:")
    print("(Press Enter to use defaults)")

    collections = input("Collections (default: sentinel-2-l2a): ").strip() or "sentinel-2-l2a"
    collections = [collections] if isinstance(collections, str) else collections

    date_range = input("Date range (YYYY-MM-DD/YYYY-MM-DD, default: 2024-06-01/2024-06-07): ").strip()
    if not date_range:
        date_range = "2024-06-01T00:00:00Z/2024-06-07T23:59:59Z"
    else:
        # Convert simple date format to ISO format
        if "/" in date_range and "T" not in date_range:
            start, end = date_range.split("/")
            date_range = f"{start}T00:00:00Z/{end}T23:59:59Z"

    try:
        limit = int(input("Limit (default: 50): ").strip() or "50")
    except ValueError:
        limit = 50

    output_dir = input(
        f"Output directory (default: {Path.home() / 'Downloads' / 'earthdaily_custom_search'}): "
    ).strip()
    if not output_dir:
        output_dir = Path.home() / "Downloads" / "earthdaily_custom_search"

    print("\n📋 Configuration summary:")
    print(f"   Collections: {collections}")
    print(f"   Date range: {date_range}")
    print(f"   Limit: {limit}")
    print(f"   Output directory: {output_dir}")

    proceed = input("\nProceed with this configuration? (y/N): ").strip().lower()
    if proceed != "y":
        print("❌ Demo cancelled")
        return

    # Execute bulk search
    client = initialize_client()
    job_id = create_bulk_search_job(client, collections, date_range, limit)

    if job_id:
        completed_job = monitor_job_status(client, job_id, max_wait_time=300)
        if completed_job:
            download_job_results(completed_job, output_dir)


def main():
    """Main function to demonstrate bulk search workflows."""
    try:
        print("🌍 EarthDaily Bulk Search Examples")
        print("=" * 60)
        print("\nThis example demonstrates bulk search capabilities.")
        print("Choose a demo to run:\n")
        print("1. Small bulk search demo (GOES collection, limited results)")
        print("2. Custom bulk search configuration")
        print("3. Run both demos")

        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == "1":
            demo_small_bulk_search()
        elif choice == "2":
            demo_custom_bulk_search()
        elif choice == "3":
            demo_small_bulk_search()
            demo_custom_bulk_search()
        else:
            print("❌ Invalid choice. Running small demo by default...")
            demo_small_bulk_search()

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("💡 Make sure to install with platform support:")
        print("   pip install 'earthdaily[platform]'")
    except Exception as e:
        print(f"\n💥 Unexpected error in main: {e}")
        print("\n💡 Make sure you have set your EDS credentials as environment variables:")
        print("   EDS_CLIENT_ID, EDS_SECRET, EDS_AUTH_URL, EDS_API_URL")


if __name__ == "__main__":
    main()
