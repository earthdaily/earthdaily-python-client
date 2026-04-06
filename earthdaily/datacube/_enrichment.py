from __future__ import annotations

from typing import Any

import rasterio
from pystac import Item
from rasterio.transform import from_gcps

from earthdaily._eds_logging import LoggerConfig
from earthdaily.datacube._builder import _replace_item_hrefs
from earthdaily.datacube.constants import DEFAULT_HREF_PATH

logger = LoggerConfig(logger_name=__name__).get_logger()


class RasterMetadataEnricher:
    """Enriches STAC items with projection and raster band metadata by reading asset files via rasterio."""

    @staticmethod
    def enrich_item(
        item: Item,
        *,
        force: bool = False,
    ) -> Item:
        """
        Enrich a STAC item with projection and raster band metadata.

        Opens each GeoTIFF asset with rasterio to extract CRS, transform, shape,
        and per-band information, then writes it back into the STAC item structure.

        Parameters
        ----------
        item : Item
            STAC item to enrich. The input item is not mutated.
        force : bool, default False
            If False, skip assets that already carry projection/raster metadata.
            If True, overwrite existing values.

        Returns
        -------
        Item
            A **new** item with enriched metadata.
        """
        item_dict = item.to_dict(transform_hrefs=False)
        resolved = _replace_item_hrefs([Item.from_dict(item_dict)], DEFAULT_HREF_PATH)
        item_dict = resolved[0].to_dict(transform_hrefs=False)

        RasterMetadataEnricher._enrich_item(item_dict, force)

        return Item.from_dict(item_dict)

    @staticmethod
    def _enrich_item(item: dict[str, Any], force: bool) -> None:
        all_gsd_values: list[float] = []
        all_epsg_values: list[int] = []
        all_epsg_code_values: list[str] = []

        current_assets: dict[str, dict] = item.get("assets", {})
        for asset_key, asset in current_assets.items():
            if not asset.get("type", "").startswith("image/tiff") or "href" not in asset:
                continue

            has_projection = bool(asset.get("proj:code") or asset.get("proj:epsg") or asset.get("proj:wkt2"))
            has_raster = bool(asset.get("raster:bands"))

            if force or not has_projection or not has_raster:
                result = RasterMetadataEnricher._process_asset(asset)
                if result is None:
                    continue

                proj_metadata, raster_metadata = result
                if proj_metadata is None:
                    logger.warning("CRS retrieval failed for item %s, asset %s.", item.get("id"), asset_key)
                    continue

                raster_data = None if not force and has_raster else raster_metadata
                RasterMetadataEnricher._update_asset_metadata(asset, proj_metadata, raster_data, force)
            else:
                RasterMetadataEnricher._copy_properties_to_asset(asset, item.get("properties", {}))

            gsd = asset.get("gsd")
            epsg = asset.get("proj:epsg")
            epsg_code = asset.get("proj:code")
            if gsd:
                all_gsd_values.append(gsd)
            if epsg:
                all_epsg_values.append(epsg)
            if epsg_code:
                all_epsg_code_values.append(epsg_code)

        RasterMetadataEnricher._update_item_properties(
            item, all_gsd_values, all_epsg_values, all_epsg_code_values, force
        )

    @staticmethod
    def _extract_metadata(dataset: Any) -> tuple[dict | None, dict | None]:
        crs = dataset.crs
        transform = dataset.transform

        if crs is None:
            gcps, gcp_crs = dataset.gcps
            if gcps and gcp_crs:
                transform = from_gcps(gcps)
                crs = gcp_crs
            else:
                return None, None

        epsg = crs.to_epsg()
        if epsg:
            crs_info = {"proj:code": f"EPSG:{epsg}", "proj:epsg": epsg}
        else:
            crs_info = {"proj:wkt2": crs.to_wkt()}

        proj_metadata = {
            "proj:transform": [
                float(transform.a),
                float(transform.b),
                float(transform.c),
                float(transform.d),
                float(transform.e),
                float(transform.f),
            ],
            "proj:shape": [dataset.height, dataset.width],
            "gsd": abs(transform.a),
            **crs_info,
        }

        bands = []
        for i in range(dataset.count):
            band_data: dict[str, Any] = {
                "data_type": str(dataset.dtypes[i]),
                "spatial_resolution": abs(transform.a),
            }

            if dataset.nodata is not None:
                if isinstance(dataset.nodata, (int, float)) or dataset.nodata in ("nan", "inf", "-inf"):
                    band_data["nodata"] = dataset.nodata

            if hasattr(dataset, "units") and dataset.units and i < len(dataset.units) and dataset.units[i]:
                band_data["unit"] = dataset.units[i]

            if (
                hasattr(dataset, "scales")
                and dataset.scales
                and i < len(dataset.scales)
                and dataset.scales[i] is not None
            ):
                band_data["scale"] = dataset.scales[i]

            if (
                hasattr(dataset, "offsets")
                and dataset.offsets
                and i < len(dataset.offsets)
                and dataset.offsets[i] is not None
            ):
                band_data["offset"] = dataset.offsets[i]

            bands.append(band_data)

        raster_metadata = {"raster:bands": bands}
        return proj_metadata, raster_metadata

    @staticmethod
    def _process_asset(asset: dict[str, Any]) -> tuple[dict | None, dict | None] | None:
        try:
            with rasterio.open(asset["href"]) as dataset:
                return RasterMetadataEnricher._extract_metadata(dataset)
        except Exception as e:
            logger.warning("Failed to open asset %s: %s", asset.get("href", "<no href>"), e)
            return None

    @staticmethod
    def _update_asset_metadata(
        asset: dict[str, Any],
        proj_metadata: dict | None,
        raster_metadata: dict | None,
        force: bool,
    ) -> None:
        if proj_metadata:
            for key, value in proj_metadata.items():
                if key not in asset or force:
                    asset[key] = value

        if raster_metadata and ("raster:bands" not in asset or force):
            asset.update(raster_metadata)

    @staticmethod
    def _copy_properties_to_asset(asset: dict[str, Any], item_properties: dict[str, Any]) -> None:
        proj_attrs = ["proj:code", "proj:epsg", "proj:wkt2", "proj:transform", "proj:shape", "gsd"]
        for attr in proj_attrs:
            if attr in item_properties and attr not in asset:
                asset[attr] = item_properties[attr]

        if "raster:bands" in item_properties and "raster:bands" not in asset:
            asset["raster:bands"] = item_properties["raster:bands"]

    @staticmethod
    def _update_item_properties(
        item: dict[str, Any],
        all_gsd_values: list[float],
        all_epsg_values: list[int],
        all_epsg_code_values: list[str],
        force: bool,
    ) -> None:
        properties = item.get("properties", {})

        unique_gsd = list(set(filter(None, all_gsd_values)))
        unique_epsg = list(set(filter(None, all_epsg_values)))
        unique_code = list(set(filter(None, all_epsg_code_values)))

        if len(unique_gsd) == 1:
            if "gsd" not in properties or properties["gsd"] is None or force:
                properties["gsd"] = unique_gsd[0]

        if len(unique_code) == 1:
            if "proj:code" not in properties or properties["proj:code"] is None or force:
                properties["proj:code"] = unique_code[0]

        if len(unique_epsg) == 1:
            if "proj:epsg" not in properties or properties["proj:epsg"] is None or force:
                properties["proj:epsg"] = unique_epsg[0]
