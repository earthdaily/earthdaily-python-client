# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.9] - 2025-04-09

### Added

- `iw-vv` and `iw-vh` short asset name for `sentinel-1-grd`.
- `ew-vv` and `ew-vh` short asset name for `sentinel-1-grd`.

### Fixed

- `cloud_mask` asset name in auto masking.

## [0.5.8] - 2025-04-08

### Added

- `polars`  method in `zonal_stats`.

### Fixed

- Naming of landsat native cloudmask.
- Supports "." in asset name for asset query.

## [0.5.7] - 2025-03-28

### Added

- `datacube` drop duplicated items and keep first item per default.

### Changed

- `deduplicate_items` renamed to `drop_duplicates` in search,
now supports "first" or "last". Default is `False` (means don't drop anything).

### Fixed

- Requirement forces `dask<2025.3.0` due to odc-stac wrong import of `quote`.
- Issue when `resampling` and non-native cloudmask in datacube.
- Issue with `rescale` when different scale/offset over time.

## [0.5.6] - 2025-03-26

### Added

- platform and python version information to User-Agent headers.

## [0.5.5] - 2025-03-11

### Fixed

- better stability of the deduplicate items algorithm.
- rescale dataset now supports same datetime in dataset.

## [0.5.4] - 2025-03-10

### Added

- `ini` credentials from `~/.earthdaily/credentials` now supports profile.

### Changed

- `filter_duplicate_items` has a time delta of 60 minutes 
(due to issues with `sentinel-2-l2a` duplicates).


## [0.5.3] - 2025-02-13

### Fixed

- `preserve_columns` in `zonal_stats`.

## [0.5.2] - 2025-02-13

### Added

- New default method for duplicates detection using `proj:transform` instead
of bbox (huge gap). Change can be done using 
`filter_duplicate_items(...,method='bbox')`.

### Fixed

- Issues with rescale when same datetime between products.

## [0.5.1] - 2025-02-12

### Added

- `deduplicate_items` for search in order to avoid several reprocessed items.
- `disable_known_warning` earthdaily global option that disables known 
dependencies warnings.

### Fixed

- `zonal-stats` supports even if `flox` is not available.

## [0.5.0] - 2025-02-11

### Added

- `zonal_stats` now preserves GeoDataFrame columns (`preserve_columns=True`).

### Changed

- `label` arg has been deprecated from `zonal_stats`.
- `zonal_stats` has same output between `xvec` and `numpy` method.
- `smart_load` is becoming `lazy_load` (`smart_load=True` is `lazy_load=False`)
- Required `pystac-client>=0.7`.
- `groupby_date` engine is fixed to `numpy`. Change is available via 
`earthdaily.option.set_option('groupby_date_engine','numba')` for example.

## [0.4.2] - 2025-02-05

### Fixed

- Parallel search issue when less than 10 days in datetime.

## [0.4.1] - 2025-02-05

### Fixed

- Parallel search issue solved when `n_jobs=-1`.

## [0.4.0] - 2025-02-03

### Added

- `native` cloudmask is now supported for `sentinel-2-c1-l2a`.
- `asset_proxy` has been implemented by @imanshafiei540, see https://github.com/earthdaily/earthdaily-python-client/issues/142

### Changed

- After benchmark, default `chunks` size is `dict(x=512,y=512, time=1)` and 
not "auto" for x and y anymore.
- `odc-stac` newest version is now supported (and fix for odc-stac has 
been submitted).

### Fixed

- Issue in rescale when several datetime were identical #146.
- Bring back the API section for the documentation #140.

## [0.3.4] - 2024-12-17

### Fixed

- `odc` stac 0.3.9 max version
- zonal_stats `wkt` has original geometry precision.
- only `bbox` in datacube search is now supported

## [0.3.3] - 2024-12-13

### Fixed

- `mode` in `zonal_stats`
- missing support for `zonal_stats` when dataset had no time dimension

### Changed

- xarray accessor classes are now privates to avoid having them in 
autocomplementation.

## [0.3.2] - 2024-12-10

### Added

- `~/.earthdaily/credentials` ini file support.

## [0.3.1] - 2024-11-22

### Fixed

- wkt geometries have now the full precision when storing it in zonal stats.

## [0.3.0] - 2024-11-19

### Added

- `ed.add_indices` accessor now supports list of strings and dict for custom
indices. For example : `['NDVI',{'NDVI2':'(nir-red)/(nir+red)'}]`.

## [0.2.15] - 2024-11-13

### Fixed

- `mask_with` list priority fixed due to new missing items warning.

## [0.2.14] - 2024-11-08

### Fixed

- parallel search import

## [0.2.13] - 2024-11-05

### Fixed

- Manage when several products have same cloudmask

## [0.2.12] - 2024-10-31

### Added

- `n_jobs` in search, default -1 (or 10).

### Fixed

- parallel search
- latitude and longitude are switched to y,x when EPSG:4326 in CRS.

## [0.2.11] - 2024-10-10

### Fixed

- `zonal_stats` now supported for python 3.10.

## [0.2.10] - 2024-10-10

### Added

- Add a retry strategy for the EarthDaily client.

### Fixed

- `bbox` for datacube has been fixed.

## [0.2.9] - 2024-09-24

### Fixed

- `prefer_alternate` is set by default to `download` in datacube is order to 
have presigned url.

## [0.2.8] - 2024-09-23

### Added

- `buffer_meters` is available in the `datacube.ed.zonal_stats` function.

### Fixed

- `bbox` argument in datacube has been restored.

## [0.2.7] - 2024-08-29

### Changed

- `intersects` in search now requests the bbox in order to optimize the request.
The items 

## [0.2.6] - 2024-08-29

### Fixed

- `intersects` computation now uses only the geometry field in order to avoid 
json parsing error.

## [0.2.5] - 2024-08-28

### Added

- `EarthDataStore` now supports a json dict in config.
- `mask_with` from datacube supports now `auto` parameters. It will ask first the
cloudmask ML, then is not available the ag_cloudmask, and at last the native one.

### Changed

- `zonal_stats` `smart_load` is replaced with `lazy_load` to be more consistant
with other libraries.

## [0.2.4] - 2024-08-13

### Changed

- `zonal_stats` now uses homemade `numpy` method. `smart_load` is set to 
default when using accessor. It loads the maximum of time (time dimension)
in memory in order to speed up the process. Statistics method (min, max...)
must by specified in the `reducers` argument.
- default `EDS_API_URL` from [https://api.eds.earthdaily.com/archive/v1/stac/v1](https://api.eds.earthdaily.com/archive/v1/stac/v1) to 
[https://api.earthdaily.com/platform/v1/stac](https://api.earthdaily.com/platform/v1/stac)


## [0.2.3] - 2024-07-23

### Added

- New method to connect to earthdatastore via `earthdaily.EarthDataStore()`
- `earthdaily.EarthDataStore()` method also supports toml/json.
- command line `copy-earthdaily-credentials-template`
- `request_payer` boolean in `EarthDataStore`method.

### Fixed

- `sentinel-1-grd` datacube wasn't generated due to a dot in the asset name.

## [0.2.2] - 2024-07-03

### Changed

- Fix odc-stac to version <= 0.3.9 due to error in auto chunks.

## [0.2.1] - 2024-06-24

### Changed

- `search` is now parallelized using datetime.

## [0.2.0] - 2024-06-19

### Added

- `datacube` `mask_with` parameters supports now `cloud_mask` (cnn cloudmask from EDA).

### Fixed 

- items `Datetime` are automatically converted in nanoseconds.
- AssetMapper is now also in the search part for only requesting the needed assets.

## [0.1.7] - 2024-06-14

### Fixed

- datacube cloud masked when using `groupby_date`.

### Changed

- accessors are now more compatible for dataarray and dataset.
- `ed.clip` accessor now manages first bounding box then clip geometry to be faster.
- New whittaker function that is about 20 times faster.
Beware, lambda from previous whittaker must be multiplied by 10 000 to have same results.

## [0.1.6] - 2024-06-05

### Fixed

- Better management of mixing several cloud cover a same day to ensure highest clear coverage.

## [0.1.5] - 2024-06-05

### Fixed

- Better management of cloud mask and error messages.
- `groupby_date` is now performed after processing the cloudmask in order to ensure a better compatibility.

## [0.1.4] - 2024-05-23

### Fixed

- type str/list for `datacube` `mask_with` kwarg.

### Added

- unit tests for mask.


## [0.1.3] - 2024-05-22

### Changed

- `mask_with` now supports list. If so, if the first mask type is not available for a collection, it will switch to the second mask.

## [0.1.2] - 2024-05-15

### Fixed

- Wrong clear_percent/clear_pixels when having multiple geometries in datacube request

## [0.1.1] - 2024-05-13

### Fixed

- `properties` of datacube is now a boolean or a list, not None by default.
- Missing crs after whittaker.


## [0.1.0] - 2024-04-19

### Fixed

- Scaled/asset when duplicated datetimes.
- Chunks asked in datacube are not the same as the output (due du smallest dims).
- Autoconverting list/dict coords to string for zarr compatibility

## [0.0.17] - 2024-04-15

### Fixed

- Issue when having different time size between sensor and cloudmask.

## [0.0.16] - 2024-04-15

### Fixed

- Issue with time when resample between datacube and cloudmask datacube.

## [0.0.15] - 2024-04-11

### Fixed

- `ag_cloud_mask_items` queries items per batch.


## [0.0.14] - 2024-04-11

### Fixed

- `GeoSeries` supported in GeometryManager.
- `ed.sel_nearest_dates` accessor avoid duplicated times.
- Issue when managing multiple indices with accessor `xr.ed`.
- Issue when same datetime when rescaling dataset.

### Added

- `mode` for zonal stats in `operations.reducers`.

### Changed

- `ag_cloud_mask_items` queries items per batch.

### Removed

- Typing decorator, expected new typing library.

## [0.0.13] - 2024-03-06

### Fixed

- `_typer` with kwargs to other functions.

## [0.0.12] - 2024-03-06

### Added

- `_typer` supports custom types.

### Changed

- `zonal_stats` outputs only valid geometries. Set `True`.
to `raise_missing_geometry` to have the old behavior.
- `zonal_stats` with geocube now manages `all_touched`.

## [0.0.11] - 2024-03-06

### Fixed

- some issues with `_typer` from accessor.
- `zonal_stats` manages index independently from row.

## [0.0.10] - 2024-03-05

### Added

- `ed.whittaker` adapted from pywapor github.
- `ed.zonal_stats` using new `geocube` zonal_stats engine.

### Changed

- `zonal_stats` now uses `geocube` as default processing method.

### Removed

- `ed.plot_index` deprecated, use `ed.plot_band` instead.

## [0.0.9] - 2024-02-29

### Fixed

-  `_typer` has a better args management.
-  `available_indices` returns only indices that can be computed.
 
## [0.0.8] - 2024-02-28

### Added

- better management of `col_wrap` in `ed` xarray accessor.

### Fixed

- some bugs in xarray `ed` accessor.

## [0.0.7] - 2024-02-28

### Added

- xarray `ed` accessor.

## [0.0.6] - 2024-02-23

### Fixed

- `to_wkt` for GeometryManager.

### Added

- Add docstring thanks to @piclem.
- Token support thanks to @luisageo6.

## [0.0.5] - 2024-02-19

### Fixed

- `groupby_date=None` is also transferred to `mask_with`.
- `mask_with="native"` also works when no assets is given.
- `clear_cover` is greater or equal (and not anymore greater).

### Added

- `properties` parameters for datacube creation.

## [0.0.4] - 2024-01-24
 
### Fixed

- `collections` from datacube now supports in args.

### Changed

- Modify dims attributs to sizes for xarray.

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
