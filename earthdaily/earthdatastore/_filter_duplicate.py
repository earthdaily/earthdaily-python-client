# mypy: ignore-errors
# TODO (v1): Fix type issues and remove 'mypy: ignore-errors' after verifying non-breaking changes

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple

from pystac import ItemCollection


@dataclass(frozen=True)
class StacGroup:
    """
    Immutable container for STAC items sharing spatial and temporal characteristics.

    Attributes
    ----------
    datetime : datetime
        Reference datetime for the group
    footprint : tuple[float, ...] | None
        Bounding box coordinates as (west, south, east, north) or None
    """

    datetime: datetime
    footprint: Optional[Tuple[float, ...]]

    def matches(
        self,
        other_dt: datetime,
        other_footprint: Optional[Tuple[float, ...]],
        threshold: timedelta,
    ) -> bool:
        """
        Check if another datetime and footprint match this group within threshold.

        Parameters
        ----------
        other_dt : datetime
            Datetime to compare against
        other_fontprint : tuple[float, ...] | None
            Proj:transform or Bounding box to compare against
        threshold : timedelta
            Maximum allowed time difference

        Returns
        -------
        bool
            True if the other datetime and bbox match within threshold
        """
        return (
            self.footprint == other_footprint
            and abs(self.datetime - other_dt) <= threshold
        )


@lru_cache(maxsize=1024)
def _parse_datetime(dt_str: str) -> datetime:
    """
    Parse ISO format datetime string to datetime object with caching.

    Parameters
    ----------
    dt_str : str
        ISO format datetime string

    Returns
    -------
    datetime
        Parsed datetime object
    """
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def _extract_item_metadata(
    item: Dict, method="proj:transform"
) -> Tuple[datetime, Optional[Tuple[float, ...]]]:
    """
    Extract datetime and footprint from a STAC item.

    Parameters
    ----------
    item : dict
        STAC item dictionary

    Returns
    -------
    tuple[datetime, tuple[float, ...] | None]
        Tuple of (datetime, footprint)
    """
    dt = _parse_datetime(item["properties"]["datetime"])
    any_asset = item.get("assets", {}) != {}
    if any_asset and method == "proj:transform":
        _first_item = list(sorted(item["assets"].keys()))[0]
        footprint = tuple(
            item["assets"][_first_item].get("proj:transform", item["bbox"])
        )
    else:
        footprint = tuple(round(bbox, 6) for bbox in item["bbox"])
    return dt, footprint


def _group_items(
    items: Iterator[Dict], time_threshold: timedelta
) -> Dict[StacGroup, List[Dict]]:
    """
    Group STAC items by spatial and temporal proximity.

    Parameters
    ----------
    items : iterator[dict]
        Iterator of STAC item dictionaries
    time_threshold : timedelta
        Maximum time difference to consider items as duplicates

    Returns
    -------
    dict[StacGroup, list[dict]]
        Dictionary mapping groups to their constituent items
    """
    groups: Dict[StacGroup, List[Dict]] = {}

    for item in items:
        dt, footprint = _extract_item_metadata(item)

        # Find matching group or create new one
        matching_group = next(
            (group for group in groups if group.matches(dt, footprint, time_threshold)),
            StacGroup(datetime=dt, footprint=footprint),
        )

        groups.setdefault(matching_group, []).append(item)

    return groups


def _deduplicate_items(items: List[Dict], keep="first") -> List[Dict]:
    """
    Select items with the most recent update timestamp.

    Parameters
    ----------
    items : list[dict]
        List of STAC items

    keep : str
        "first" or "last".

    Returns
    -------
    list[dict]
        Items with the latest update timestamp
    """
    if not items:
        return []
    if len(items) == 1:
        return items

    keep_idx = {"first": 0, "last": -1}
    item = [sorted(items, key=lambda x: x["id"])[keep_idx[keep]]]
    return item


def filter_duplicate_items(
    items: ItemCollection,
    time_threshold: timedelta = timedelta(minutes=60),
    method="proj:transform",
    keep="first",
) -> ItemCollection:
    """
    Deduplicate STAC items based on spatial and temporal proximity.

    This function groups items by their bounding box and temporal proximity,
    then selects the latest version(s) from each group based on their update timestamp.

    Parameters
    ----------
    items : ItemCollection
        Collection of STAC items to deduplicate. Each item must have:
            - properties.datetime : str
                ISO format acquisition timestamp
            - properties.updated : str
                ISO format update timestamp
            - footprint : list[float], optional
                Bounding box coordinates [west, south, east, north] or proj:transform.

    time_threshold : timedelta, optional
        Maximum time difference to consider items as duplicates, by default 5 minutes

    method : str
        'proj:transform' or 'bbox' to compare against

    Returns
    -------
    ItemCollection
        Deduplicated collection containing only the latest versions of each item group

    Notes
    -----
    Items are considered duplicates if they:
        1. Share the same bounding box (or both lack a bounding box)
        2. Have acquisition timestamps within the specified threshold
    From each group of duplicates, all items sharing the latest update timestamp
    are retained.

    Examples
    --------
    >>> from datetime import timedelta
    >>> items = ItemCollection([{
    ...     "id": "S2A_31TCJ_20190416_0_L2A",
    ...     "bbox": [1, 2, 3, 4],
    ...     "properties": {
    ...         "datetime": "2019-04-16T10:02:30Z",
    ...         "updated": "2019-04-16T10:00:00Z"
    ...     }
    ... }, {
    ...     "id": "S2A_31TCJ_20190416_1_L2A",
    ...     "bbox": [1, 2, 3, 4],
    ...     "properties": {
    ...         "datetime": "2019-04-16T10:02:45Z",
    ...         "updated": "2019-04-16T11:00:00Z"
    ...     }
    ... }])
    >>> result = deduplicate_items(items, timedelta(minutes=1))
    >>> len(result)
    1
    >>> result[0]["id"]
    'S2A_31TCJ_20190416_1_L2A'
    """
    # Convert ItemCollection to features and group them
    grouped_items = _group_items(items.to_dict()["features"], time_threshold)

    # Select latest items from each group and flatten
    deduplicated = [
        item
        for group_items in grouped_items.values()
        for item in _deduplicate_items(group_items, keep=keep)
    ]
    duplicates = len(items) - len(deduplicated)
    if duplicates:
        logging.info(f"Deduplication removes {duplicates} items.")

    deduplicated = sorted(deduplicated, key=lambda x: x["properties"]["datetime"])

    return ItemCollection(deduplicated)
