from typing import Any

from pystac import Item

from earthdaily.datacube._builder import _deduplicate_items, build_datacube
from earthdaily.datacube._datacube import Datacube
from earthdaily.datacube.constants import DEFAULT_DTYPE, DEFAULT_ENGINE, DEFAULT_HREF_PATH, DEFAULT_NODATA


class DatacubeService:
    def create(
        self,
        items: list[Item],
        assets: list[str] | dict[str, str] | None = None,
        bbox: list[float] | tuple | None = None,
        intersects: Any = None,
        dtype: str = DEFAULT_DTYPE,
        nodata: float | int | None = DEFAULT_NODATA,
        properties: bool | str | list[str] = False,
        engine: str = DEFAULT_ENGINE,
        replace_href_with: str = DEFAULT_HREF_PATH,
        deduplicate_by: list[str] | None = None,
        deduplicate_keep: str = "last",
        **kwargs,
    ) -> Datacube:
        if deduplicate_by:
            items = _deduplicate_items(items, deduplicate_by, deduplicate_keep)

        dataset = build_datacube(
            items=items,
            assets=assets,
            bbox=bbox,
            intersects=intersects,
            dtype=dtype,
            nodata=nodata,
            properties=properties,
            engine=engine,
            replace_href_with=replace_href_with,
            **kwargs,
        )

        metadata = {"items_count": len(items), "engine": engine}

        return Datacube(dataset, metadata)
