"""
SIMEDC Analysis-Ready Data Datacube Example
============================================

Demonstrates datacube capabilities with the simedc-analysis-ready-data collection:
- Creation with SIMEDC-specific assets
- Quality mask application
- Spectral indices using red edge bands
- Temporal operations
- Visualization

Requirements:
- pip install 'earthdaily[datacube]'
- Set EDS credentials as environment variables
"""

import math
import os
import warnings

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from rasterio.errors import NotGeoreferencedWarning

from earthdaily import EDSClient, EDSConfig

os.environ["GDAL_HTTP_MAX_RETRY"] = "10"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

COLLECTION_ID = "simedc-analysis-ready-data"
DATE_RANGE = "2022-12-01T00:00:00Z/2022-12-31T23:59:59Z"
SEARCH_BBOX = [77.5, 26.7, 78.1, 28.0]
MAX_ITEMS = 3
CLOUD_COVER_QUERY = {"eo:cloud_cover": {"lt": 30}}
SPECTRAL_ASSETS = [
    "image_file_B",
    "image_file_G",
    "image_file_R",
    "image_file_NIR",
    "image_file_RE1",
    "image_file_RE2",
    "image_file_RE3",
]
QUALITY_MASK_ASSETS = ["quality_mask"]
TARGET_RESOLUTION = 10
RECHUNK_CONFIG = {"x": 512, "y": 512, "time": 1}
TEMPORAL_AGG_FREQ = "1ME"
RESAMPLE_FREQ = "7D"
WHITTAKER_BETA = 10000
WHITTAKER_MIN_DAYS = 3
VISUALIZATION_MAX_DIM = 4000


def _bbox_to_polygon(bbox):
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [minx, maxy],
                [maxx, maxy],
                [maxx, miny],
                [minx, miny],
            ]
        ],
    }


def _scaled_band(data_array):
    def _first(keys, default):
        for key in keys:
            value = data_array.attrs.get(key)
            if value is not None:
                return value
        return default

    scale = _first(("scale_factor", "scale", "scaling_factor"), 1.0)
    offset = _first(("add_offset", "offset"), 0.0)
    try:
        scale_value = float(scale)
    except (TypeError, ValueError):
        scale_value = 1.0
    try:
        offset_value = float(offset)
    except (TypeError, ValueError):
        offset_value = 0.0
    return data_array.astype("float32") * scale_value + offset_value


def _prepare_visualization_cube(cube):
    dataset = cube.data
    y_dim = next((dim for dim in ("y", "latitude", "lat", "Y") if dim in dataset.dims), None)
    x_dim = next((dim for dim in ("x", "longitude", "lon", "X") if dim in dataset.dims), None)
    if not y_dim or not x_dim:
        return cube, False
    y_size = dataset.sizes.get(y_dim, 0)
    x_size = dataset.sizes.get(x_dim, 0)
    if y_size <= VISUALIZATION_MAX_DIM and x_size <= VISUALIZATION_MAX_DIM:
        return cube, False
    y_factor = max(1, math.ceil(y_size / VISUALIZATION_MAX_DIM))
    x_factor = max(1, math.ceil(x_size / VISUALIZATION_MAX_DIM))
    if y_factor == 1 and x_factor == 1:
        return cube, False
    reduced = dataset.coarsen({y_dim: y_factor, x_dim: x_factor}, boundary="trim").mean()
    return cube.with_dataset(reduced), True


def main():
    config = EDSConfig()
    client = EDSClient(config)

    search_result = client.platform.pystac_client.search(
        collections=[COLLECTION_ID],
        datetime=DATE_RANGE,
        bbox=SEARCH_BBOX,
        query=CLOUD_COVER_QUERY,
        max_items=MAX_ITEMS,
        sortby=[{"field": "properties.datetime", "direction": "asc"}],
    )
    items = list(search_result.items())
    print(f"Found {len(items)} SIMEDC items\n")

    if not items:
        print("No items found. Please check your search parameters.")
        return

    aoi_geometry = _bbox_to_polygon(SEARCH_BBOX)

    print("Sample item assets:")
    sample_assets = list(items[0].assets.keys())
    print(f"  Available assets: {sample_assets[:10]}...")
    print()

    # Tip: Pass pool=N to enable parallel COG metadata fetching for faster loading
    # Example: client.datacube.create(..., pool=os.cpu_count())
    datacube = client.datacube.create(
        items=items,
        assets=SPECTRAL_ASSETS,
        resolution=TARGET_RESOLUTION,
        deduplicate_by=["date"],
    )
    print(f"1. Created datacube: {datacube.shape}")
    print(f"   Bands: {datacube.bands}")

    quality_mask_cube = client.datacube.create(
        items=items,
        assets=QUALITY_MASK_ASSETS,
        resolution=TARGET_RESOLUTION,
        deduplicate_by=["date"],
    )
    print(f"2. Created quality mask datacube: {quality_mask_cube.shape}")
    print(f"   Quality mask bands: {quality_mask_cube.bands}")

    mask_band_name = None
    for band in quality_mask_cube.bands:
        if "quality_mask" in band.lower() or "cloud" in band.lower():
            mask_band_name = band
            break

    if mask_band_name:
        try:
            masked = datacube.apply_mask(
                mask_band=mask_band_name,
                mask_dataset=quality_mask_cube.data,
                exclude_values=[2],
                mask_statistics=True,
            )
            print(
                f"3. Applied quality mask using '{mask_band_name}' "
                f"(excluding cloud/shadow/haze): {len(masked.timestamps)} timestamps"
            )
        except Exception as e:
            print(f"3. Masking failed: {e}. Continuing without masking.")
            masked = datacube
    else:
        print("3. Quality mask band not found in expected format, skipping masking")
        masked = datacube

    red_band = _scaled_band(masked.data["image_file_R"])
    nir_band = _scaled_band(masked.data["image_file_NIR"])
    blue_band = _scaled_band(masked.data["image_file_B"])
    ndvi_params = {
        "R": red_band,
        "N": nir_band,
    }
    evi_params = {
        "B": blue_band,
        # Provide both lowercase/uppercase names to satisfy spyndex parameter aliases.
        "g": 2.5,
        "G": 2.5,
        "c1": 6.0,
        "C1": 6.0,
        "c2": 7.5,
        "C2": 7.5,
        "L": 1.0,
        "l": 1.0,
    }

    try:
        with_indices = masked.add_indices(
            ["NDVI", "EVI"],
            **ndvi_params,
            **evi_params,
        )
        print(f"4. Added spectral indices: {with_indices.bands}")
    except Exception as e:
        print(f"4. Some indices failed: {e}. Trying with NDVI only...")
        try:
            with_indices = masked.add_indices(
                ["NDVI"],
                **ndvi_params,
            )
            print(f"   Added NDVI: {with_indices.bands}")
        except Exception as e2:
            print(f"   Index calculation failed: {e2}. Continuing without indices.")
            with_indices = masked

    if len(with_indices.timestamps) >= 2:
        times = sorted(with_indices.data.time.values)
        time_filtered = with_indices.select_time(str(times[0]), str(times[-1]))
        print(f"5. Time filtered: {len(time_filtered.timestamps)} timestamps")
    else:
        time_filtered = with_indices
        print(f"5. Time filtering skipped: only {len(with_indices.timestamps)} timestamp(s)")

    rechunked = time_filtered.rechunk(RECHUNK_CONFIG)
    print(f"6. Rechunked: {rechunked.shape}")

    print("\nDatacube Properties:")
    print(f"  bands: {rechunked.bands}")
    print(f"  timestamps: {len(rechunked.timestamps)}")
    print(f"  crs: {rechunked.crs}")
    print(f"  resolution: {rechunked.resolution}")
    print(f"  extent: {rechunked.extent}")
    print(f"  data type: {type(rechunked.data).__name__}")

    if len(rechunked.timestamps) >= 2:
        aggregated = rechunked.temporal_aggregate(method="mean", freq=TEMPORAL_AGG_FREQ)
        print(f"\n7. Temporal aggregate (monthly mean): {len(aggregated.timestamps)} timestamps")

        resampled = rechunked.resample(freq=RESAMPLE_FREQ, method="median")
        print(f"8. Resampled (7-day median): {len(resampled.timestamps)} timestamps")

        unique_times_after = len(set(rechunked.data.time.values))
        time_index = rechunked.data.indexes.get("time")
        if time_index is not None:
            unique_days = len(time_index.floor("D").unique())
        else:
            unique_days = unique_times_after
        if unique_times_after >= 2 and unique_days >= WHITTAKER_MIN_DAYS:
            try:
                whittaker_smoothed = rechunked.whittaker(beta=WHITTAKER_BETA)
                print(f"9. Whittaker smoothed: {whittaker_smoothed.shape}")
            except Exception as e:
                print(f"9. Whittaker smoothing skipped: {e}")
        else:
            print(
                "9. Whittaker smoothing skipped: need at least "
                f"{WHITTAKER_MIN_DAYS} unique observation days "
                f"(unique timestamps={unique_times_after}, unique days={unique_days})"
            )
    else:
        print("\n7-9. Temporal operations skipped: insufficient timestamps")

    try:
        merged = datacube.merge(quality_mask_cube)
        print(f"10. Merged spectral + mask bands: {merged.bands}")
    except Exception as e:
        print(f"10. Merge failed: {e}")

    try:
        clipped = rechunked.clip(aoi_geometry)
        print(f"11. Clipped: {clipped.shape}")
    except Exception as e:
        print(f"11. Clipping failed: {e}")

    try:
        zonal_result = rechunked.zonal_stats(aoi_geometry, reducers=["mean", "std", "min", "max"])
        print(f"12. Zonal stats: {type(zonal_result).__name__}")
        if hasattr(zonal_result, "columns"):
            print(f"    Statistics computed for: {list(zonal_result.columns)}")
    except Exception as e:
        print(f"12. Zonal stats failed: {e}")

    info_text = rechunked.info()
    print(f"\n13. Datacube info:\n{info_text}")
    plot_source, plot_downsampled = _prepare_visualization_cube(rechunked)
    if plot_downsampled:
        original_shape = rechunked.shape
        reduced_shape = plot_source.shape
        print(
            "   Note: downsampled visualization cube from "
            f"{original_shape.get('y', '?')}x{original_shape.get('x', '?')} to "
            f"{reduced_shape.get('y', '?')}x{reduced_shape.get('x', '?')} for plotting."
        )

    rgb_bands = ["image_file_R", "image_file_G", "image_file_B"]
    if all(band in plot_source.bands for band in rgb_bands):
        rgb_data = plot_source.data[rgb_bands]
        rgb_array = rgb_data.to_array(dim="bands")
        rgb_valid = rgb_array.where(rgb_array.notnull())
        if rgb_valid.notnull().any():
            vmin = float(rgb_valid.quantile(0.02).values)
            vmax = float(rgb_valid.quantile(0.98).values)
        else:
            vmin, vmax = 0, 1

        rgb_plot = plot_source.plot_rgb(
            red="image_file_R",
            green="image_file_G",
            blue="image_file_B",
            vmin=vmin,
            vmax=vmax,
        )
        print(f"14. RGB plot: {type(rgb_plot).__name__}")
        rgb_plot.fig.savefig("simedc_rgb_plot.png", dpi=150, bbox_inches="tight")
        print("    Saved to: simedc_rgb_plot.png")

        thumb = rechunked.thumbnail(
            red="image_file_R",
            green="image_file_G",
            blue="image_file_B",
            time_index=0,
            vmin=vmin,
            vmax=vmax,
        )
        print(f"15. Thumbnail: {type(thumb).__name__}")
        thumb.savefig("simedc_thumbnail.png", dpi=150, bbox_inches="tight")
        print("    Saved to: simedc_thumbnail.png")
    else:
        print("14-15. RGB visualization skipped: required bands not available")

    if "NDVI" in plot_source.bands:
        band_plot = plot_source.plot_band("NDVI", cmap="RdYlGn", robust=True)
        print(f"16. NDVI band plot: {type(band_plot).__name__}")
        band_plot.fig.savefig("simedc_ndvi_plot.png", dpi=150, bbox_inches="tight")
        print("    Saved to: simedc_ndvi_plot.png")

    if "EVI" in plot_source.bands:
        band_plot = plot_source.plot_band("EVI", cmap="viridis", robust=True)
        print(f"17. EVI band plot: {type(band_plot).__name__}")
        band_plot.fig.savefig("simedc_evi_plot.png", dpi=150, bbox_inches="tight")
        print("    Saved to: simedc_evi_plot.png")

    print("SIMEDC datacube example completed successfully!")


if __name__ == "__main__":
    main()
