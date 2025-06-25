# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:03:08 2023

@author: nkk
"""

import json

from earthdaily import EDSClient, EDSConfig
from earthdaily.legacy.earthdatastore.cube_utils.asset_mapper import (
    __asset_mapper_config_path,
    _asset_mapper_config,
)

if __name__ == "__main__":
    eds = EDSClient(EDSConfig())
    asset_mapper_path = __asset_mapper_config_path
    asset_mapper_config = _asset_mapper_config

    for collection in eds.legacy.explore():
        try:
            assets_name = list(eds.legacy.explore(collection).item.assets.keys())
        except AttributeError:
            print(f"collection {collection} has no items")
            continue
        for asset_name in assets_name:
            if asset_mapper_config.get(collection) is None:
                asset_mapper_config[collection] = [{}]
            if asset_name not in asset_mapper_config[collection][0].keys():
                asset_mapper_config[collection][0][asset_name] = asset_name

    with open(asset_mapper_path, "w", encoding="utf-8") as f:
        json.dump(asset_mapper_config, f, ensure_ascii=False, indent=4)
