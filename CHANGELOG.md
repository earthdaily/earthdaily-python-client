# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1-rc5]

### Added

- `clear_cover` in the datacube method when using the `earthdatastore.Auth` method.

### Changed

- masks statistics are not anymore suffixed with the cloudmask type : `clear_percent`and `clear_pixels`. Warns with a DeprecationWarning.
- all queries in `post_query` must return True to keep the item. If a key doesn't exist, considers the result as False (instead of failling).

### Fixed

- search `post_query` do not block if some properties are not available on all items.

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

- Python version must be at least 3.10
- Gdal version must be at least 3.7.0
- A better documentation


## [0.0.1-beta] 2023-09-20

### Added 

- First public release under the name of earthdaily
- module earthdatastore gathers all cube fonctions
- earthdatastore.Auth() allows methods for search and datacube creation
- Add if missing default scale/offset factor to collection (landsat-c2l2-sr, landsat-c2l2-st, landsat-c2l1)