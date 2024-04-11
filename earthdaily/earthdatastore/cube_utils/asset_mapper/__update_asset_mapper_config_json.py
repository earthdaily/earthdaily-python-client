# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:03:08 2023

@author: nkk
"""

import earthdaily
import json

eds = earthdaily.earthdatastore.Auth()
asset_mapper_path = (
    earthdaily.earthdatastore.cube_utils.asset_mapper.__asset_mapper_config_path
)
asset_mapper_config = (
    earthdaily.earthdatastore.cube_utils.asset_mapper._asset_mapper_config
)

for collection in eds.explore():
    try:
        assets_name = list(eds.explore(collection).item.assets.keys())
    except AttributeError:
        print(f"collection {collection} has no items")
        continue
    for asset_name in assets_name:
        print(asset_name)
        if asset_mapper_config.get(collection) is None:
            asset_mapper_config[collection] = [{}]
        if asset_name not in asset_mapper_config[collection][0].values():
            asset_mapper_config[collection][0][asset_name] = asset_name

with open(asset_mapper_path, "w", encoding="utf-8") as f:
    json.dump(asset_mapper_config, f, ensure_ascii=False, indent=4)
