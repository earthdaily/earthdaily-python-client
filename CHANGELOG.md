# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1-rc2] 


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