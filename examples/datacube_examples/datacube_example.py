"""
Datacube Module Example
========================

Demonstrates ALL datacube capabilities:
- Creation, masking, indices, spatial ops, temporal ops, visualization, properties

Requirements:
- pip install 'earthdaily[datacube]'
- Set EDS credentials as environment variables
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

from earthdaily import EDSClient, EDSConfig


def main():
    config = EDSConfig()
    client = EDSClient(config)

    search_result = client.platform.pystac_client.search(
        collections=["sentinel-2-l2a"],
        datetime="2024-07-01T00:00:00Z/2024-07-15T23:59:59Z",
        bbox=[-122.5, 37.7, -122.3, 37.9],
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=2,
    )
    items = list(search_result.items())
    print(f"Found {len(items)} items\n")

    if not items:
        return

    datacube = client.datacube.create(
        items=items,
        assets=["red", "green", "blue", "nir", "scl"],
        resolution=60,
    )
    print(f"1. Created: {datacube.shape}")

    masked = datacube.apply_mask(mask_band="scl", exclude_values=[3, 8, 9, 10], mask_statistics=True)
    print(f"2. Masked: {len(masked.timestamps)} timestamps")

    with_indices = masked.add_indices(["NDVI"], R=masked.data["red"], N=masked.data["nir"])
    print(f"3. Added indices: {with_indices.bands}")

    if len(with_indices.timestamps) >= 2:
        times = sorted(with_indices.data.time.values)
        time_filtered = with_indices.select_time(str(times[0]), str(times[-1]))
        print(f"4. Time filtered: {len(time_filtered.timestamps)} timestamps")
    else:
        time_filtered = with_indices
        print(f"4. Time filtering skipped: only {len(with_indices.timestamps)} timestamp(s)")

    rechunked = time_filtered.rechunk({"x": 512, "y": 512, "time": 1})
    print(f"5. Rechunked: {rechunked.shape}")

    print("\nProperties:")
    print(f"  bands: {rechunked.bands}")
    print(f"  timestamps: {len(rechunked.timestamps)}")
    print(f"  crs: {rechunked.crs}")
    print(f"  resolution: {rechunked.resolution}")
    print(f"  extent: {rechunked.extent}")
    print(f"  data: {type(rechunked.data).__name__}")

    aggregated = rechunked.temporal_aggregate(method="mean", freq="1ME")
    print(f"6. Temporal aggregate: {len(aggregated.timestamps)} timestamps")

    resampled = rechunked.resample(freq="7D", method="median")
    print(f"7. Resampled: {len(resampled.timestamps)} timestamps")

    whittaker_smoothed = rechunked.whittaker(beta=10000)
    print(f"8. Whittaker smoothed: {whittaker_smoothed.shape}")

    merged = rechunked.merge(rechunked)
    print(f"9. Merged: {merged.bands}")

    extent = rechunked.extent
    if extent:
        x_min, y_min, x_max, y_max = extent
        x_margin = (x_max - x_min) * 0.2
        y_margin = (y_max - y_min) * 0.2

        geometry_shape = box(x_min + x_margin, y_min + y_margin, x_max - x_margin, y_max - y_margin)
        gdf = gpd.GeoDataFrame([1], geometry=[geometry_shape], crs=rechunked.crs)
        geometry = gdf.to_crs("EPSG:4326").geometry.iloc[0].__geo_interface__

        clipped = rechunked.clip(geometry)
        print(f"10. Clipped: {clipped.shape}")

        zonal_result = rechunked.zonal_stats(geometry, reducers=["mean", "std"])
        print(f"11. Zonal stats: {type(zonal_result).__name__}")
    else:
        print("10. Clipping skipped: no extent available")
        print("11. Zonal stats skipped: no extent available")

    info_text = rechunked.info()
    print(f"12. Info: {info_text}")

    rgb_data = rechunked.data[["red", "green", "blue"]]
    rgb_array = rgb_data.to_array(dim="bands")
    rgb_valid = rgb_array.where(rgb_array.notnull())
    if rgb_valid.notnull().any():
        vmin = float(rgb_valid.quantile(0.02).values)
        vmax = float(rgb_valid.quantile(0.98).values)
    else:
        vmin, vmax = 0, 1

    rgb_plot = rechunked.plot_rgb(red="red", green="green", blue="blue", vmin=vmin, vmax=vmax)
    print(f"13. RGB plot: {type(rgb_plot).__name__}")
    rgb_plot.fig.savefig("rgb_plot.png", dpi=150, bbox_inches="tight")
    print("    Saved to: rgb_plot.png")
    plt.close(rgb_plot.fig)

    band_plot = rechunked.plot_band("NDVI", cmap="RdYlGn", robust=True)
    print(f"14. Band plot (NDVI): {type(band_plot).__name__}")
    band_plot.fig.savefig("band_plot.png", dpi=150, bbox_inches="tight")
    print("    Saved to: band_plot.png")
    plt.close(band_plot.fig)

    thumb = rechunked.thumbnail(red="red", green="green", blue="blue", time_index=0, vmin=vmin, vmax=vmax)
    print(f"15. Thumbnail: {type(thumb).__name__}")
    thumb.savefig("thumbnail.png", dpi=150, bbox_inches="tight")
    print("    Saved to: thumbnail.png")
    plt.close(thumb)


if __name__ == "__main__":
    main()
