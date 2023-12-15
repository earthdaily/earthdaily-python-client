import os
import json

__pathFile = os.path.dirname(os.path.realpath(__file__))
__asset_mapper_config_path = os.path.join(__pathFile, "asset_mapper_config.json")
_asset_mapper_config = json.load(open(__asset_mapper_config_path))


class AssetMapper:
    def __init__(self):
        self.available_collections = list(_asset_mapper_config.keys())

    def collection_mapping(self, collection):
        if self._collection_exists(collection, raise_warning=True):
            return _asset_mapper_config[collection]

    def _collection_exists(self, collection, raise_warning=False):
        exists = True if collection in self.available_collections else False
        if raise_warning and not exists:
            raise NotImplementedError(
                f"Collection {collection} has not been implemented"
            )
        return exists

    def collection_spectral_assets(self, collection):
        return self.collection_mapping(collection)

    def map_collection_assets(self, collection, assets):
        if isinstance(assets, (dict | None)):
            return assets
        if not self._collection_exists(collection):
            return assets

        # HANDLE LIST TO DICT CONVERSION
        if isinstance(assets, list):
            assets = {asset: asset for asset in assets}

        output_assets = {}

        config = self.collection_mapping(collection)

        # Try to map each asset
        for asset in assets:
            if asset in config[0]:
                output_assets[config[0][asset]] = asset
            # No asset found with specified key (common asset name)
            else:
                # Looking for asset matching the specified value (asset name)
                matching_assets = [
                    key for key, value in config[0].items() if value == asset
                ]

                if matching_assets:
                    output_assets[asset] = asset
        return output_assets
