# EarthDaily Datacube Module

A modern, flexible datacube module for creating and manipulating multi-dimensional geospatial datasets from STAC items.

## Installation

```bash
pip install 'earthdaily[datacube]'
```

## Quick Start

```python
from earthdaily import EDSClient, EDSConfig

config = EDSConfig()
client = EDSClient(config)

items = client.platform.bulk_search.search(
    collections=["sentinel-2-l2a"],
    intersects=geometry,
    datetime=["2023-06-01", "2023-08-01"],
)

datacube = client.datacube.create(
    items=items,
    assets=["red", "green", "blue", "nir"],
    resolution=10
)
```

## Available Capabilities

### 1. Creation
- `client.datacube.create(items, assets, resolution)` - Create datacubes from STAC items using odc-stac

If `boto3` is installed and datacube loads fail with AWS credential errors, set `AWS_NO_SIGN_REQUEST=YES`. Rasterio will then skip the AWS credential chain for public and presigned objects.

### 2. Cloud Masking
- `datacube.apply_mask(mask_band, clear_values, exclude_values)` - Apply cloud/quality masks with statistics
- Supports custom mask functions and automatic clear cover calculation

### 3. Spatial Operations
- `datacube.clip(geometry)` - Clip to spatial extent (supports GeoJSON, WKT, GeoDataFrame)
- `datacube.merge(other, compat)` - Merge with another datacube
- `datacube.rechunk(chunks)` - Optimize Dask chunking

### 4. Temporal Operations
- `datacube.select_time(start, end)` - Filter by time range, or `select_time(time=...)` to select specific acquisitions
- `datacube.temporal_aggregate(method, freq)` - Aggregate over time (mean, median, min, max, sum, std, var)
- `datacube.resample(freq, method)` - Resample to different frequency
- `datacube.whittaker(beta, weights)` - Temporal smoothing (Whittaker filter)

### 5. Spectral Indices
- `datacube.add_indices(["NDVI", "EVI", ...], R=red, N=nir)` - Add spectral indices via spyndex

### 6. Zonal Statistics
- `datacube.zonal_stats(geometry, reducers)` - Compute statistics per geometry zone
- Reducers: mean, median, min, max, sum, std, var, mode
- Returns xarray Dataset with feature and zonal_statistics dimensions

### 7. Visualization
- `datacube.plot_rgb(red, green, blue)` - RGB composite visualization
- `datacube.plot_band(band, cmap)` - Single band visualization
- `datacube.thumbnail(time_index)` - Quick preview of single timestep

### 8. Properties & Metadata
- `datacube.bands` - List of band names
- `datacube.timestamps` - List of datetime values
- `datacube.crs` - Coordinate reference system
- `datacube.resolution` - Spatial resolution (x, y)
- `datacube.shape` - Dimensions dict (e.g., `{"time": 10, "y": 100, "x": 100}`)
- `datacube.extent` - Spatial extent (minx, miny, maxx, maxy)
- `datacube.data` - Direct access to underlying xarray Dataset
- `datacube.info()` - Print comprehensive datacube information

## Write a GeoTIFF

`rioxarray` is installed with the datacube extra and adds the `.rio` accessor. This writes one multiband float GeoTIFF for a single acquisition. Write one file per band by calling `to_raster` on each data variable instead.

```python
import rioxarray  # registers the .rio accessor

ds = datacube.data

rgb_ds = ds[["red", "green", "blue"]]
if "time" in rgb_ds.dims:
    rgb_ds = rgb_ds.squeeze("time", drop=True)  # single acquisition

rgb = (
    rgb_ds.to_array(dim="band")
    .assign_coords(band=["red", "green", "blue"])
    .transpose("band", "y", "x")
)

rgb.rio.to_raster("rgb_float32.tif", dtype="float32", compress="deflate", tiled=True)
```

## Architecture

The module is organized into separate files for maintainability:

```
earthdaily/datacube/
├── __init__.py              # Module exports
├── _datacube_service.py     # DatacubeService (entry point)
├── _datacube.py             # Datacube wrapper class
├── _builder.py              # Creation logic with odc-stac
├── _geometry.py             # Geometry utilities (GeoJSON, WKT, GeoDataFrame)
├── _masking.py              # Cloud masking
├── _operations.py           # Clip, merge, select_time, rechunk
├── _indices.py              # Spectral indices via spyndex
├── _zonal.py                # Zonal statistics implementation
├── _temporal.py             # Temporal aggregation & resampling
├── _whittaker.py            # Whittaker smoothing filter
├── _visualization.py        # Plotting methods
├── models.py                # Type definitions (Literal types)
└── exceptions.py            # Custom exceptions
```

## Examples

See `examples/datacube_examples/datacube_example.py` for a complete working example.

