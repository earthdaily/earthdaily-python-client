# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4]
 
## [0.0.3] - 2023-12-15

### Added

- Ability to query assets in the search items.

## [0.0.2] - 2023-12-15

### Fixed

- Missing json due to pypi installation.
- Multisensors datacube now support `stackstac`.

## [0.0.1] - 2023-12-15

### Added

- Auth datacube `earthdaily.earthdatastore.Auth().datacube(...)` function now manages multiple collections.
- Uses `fields` to query only assets asked over Skyfox/EDS (better performance).

### Changed

- `asset_mapper` has now all assets from available collections.

## [0.0.1-rc9] - 2023-12-12

### Added

- Optimization (dask-native) for cloudmasks (Sentinel-2, Venus, ag-cloud-mask), except Landsat.

### Changed

- Default chunks using odc is now `x="auto"` and `y="auto"`.
- `geobox` or `geopolygon` is used to have a preclipped datacube to the bounding box of the geometry. It enhances performances.

### Fixed

- Load json config for `Auth`.
- Remove kwargs of `geobox` for native cloudmask in order to parse new `geobox`.

## [0.0.1-rc8] - 2023-07-12

### Added

- Tests for python 3.12.
- handle cross calibration (harmonization) using private collection.
- `Auth` automatically to EarhtDataStore after 3600 seconds.
- `Auth` now supports presign_urls parameter.
- `load_pivot_corumba` in datasets.

### Fixed

- Perfect dimension compatibility between cloudmask and sensor datacube using `geobox`.
- Force loading native landsat cloudmask because not compatible with dask.

## [0.0.1-rc7] - 2023-11-22

### Fixed

- issue with unlogged datacube and non-geodataframe intersects.
- for element84 sentinel-2-l2a collection, `boa_offset_applied`is set to True since 2022-02-28.
- added `nir` band in `boa_offset_applied` control.
- fix percentage in landsat agriculture cloudmask .

## [0.0.1-rc6] - 2023-11-10

### Fixed

- fix bug in `setup.py` using gh actions.

## [0.0.1-rc5] - 2023-11-10

### Added

- `intersects` argument in search/datacube now supports wkt/geojson/geopandas.
- `common_band_names` default set to True in datacube creation. It uses the new Assets Mapper in order to define to best suitable bands according to user needs.
- `clear_cover` argument in the datacube method when using the `earthdatastore.Auth` method.
- `datasets.load_pivot()` to load a GeoDataFrame of a pivot in Nebraska (alternates between corn or soy between years).
- `preload_mask` in authenticated datacube method set to `True`by default to load, if enough virtual memory. 
- Several tests to check and validate code.
- Better performances for cloud mask statistics by checking data type.

### Changed

- masks statistics are not anymore suffixed with the cloudmask type : `clear_percent` and `clear_pixels`. Warns with a DeprecationWarning.
- all queries in `post_query` must return True to keep the item. If a key doesn't exist, considers the result as False (instead of failling).
- default chunks are now `x=512` and `y=512` for odc-stac.

### Fixed

- search `post_query` do not block if some properties are not available on all items.
- some scale/offsets were not supported due to missing scale/offsets from previous assets.
- issues when computing datacube using Landsat cloudmask (`qa_pixel`).
- `intersects` now supports several geometries and don't force selection on the first index.


## [0.0.1-rc4] 2023-10-19

### Changed

- When asking `ag_cloud_mask`, keep only sensor items that have `eda:ag_cloud_mask_available=True`.

## [0.0.1-rc3] 2023-10-18

### Added

- `earthdaily.earthdatastore.cube_utils.zonal_stats_numpy` to compute local statistics using numpy functions. Best when high number of geometries.

### Changed

- `zonal_stats` default parameters rasterizes now whole vector in order to be
really faster faster. Previous behavior is available by selecting `method="standard"`.
 
### Fixed

- Fix when number of `ag_cloud_mask` is lower than number of sensor items

## [0.0.1-rc2] 2023-10-12

### Fixed

- Add `stats` name as dimension

### Changed

- `zonal_stats` output "feature" (not "feature_name") and `stats` dimensions


## [0.0.1-rc1] 2023-10-12

### Added

- Automatic presigned url when needed, just specify argument `prefer_alternate="download"` in search.
- `zonal_stats` function available in `earthdatastore.cube_utils` module.
- An exemple on how to use `.env` credentials.

### Changed

- Python version must be at least 3.10.
- Gdal version must be at least 3.7.0.
- A better documentation.


## [0.0.1-beta] 2023-09-20

### Added 

- First public release under the name of earthdaily.
- module earthdatastore gathers all cube fonctions.
- earthdatastore.Auth() allows methods for search and datacube creation.
- Add if missing default scale/offset factor to collection (landsat-c2l2-sr, landsat-c2l2-st, landsat-c2l1).