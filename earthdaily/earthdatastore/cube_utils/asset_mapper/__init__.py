from . import _asset_mapper_config


class AssetMapper:
    def __init__(self):
        self.available_collections = list(
            _asset_mapper_config.asset_mapper_collections.keys()
        )

    def collection_mapping(self, collection):
        if self._collection_exists(collection, raise_warning=True):
            return _asset_mapper_config.asset_mapper_collections[collection]

    def _collection_exists(self, collection, raise_warning=False):
        exists = True if collection in self.available_collections else False
        if raise_warning and not exists:
            raise NotImplementedError(
                f"Collection {collection} has not been implemented"
            )
        return exists

    def map_collection_bands(self, collection, bands):
        if isinstance(bands, (dict|None)):
            return bands
        if not self._collection_exists(collection):
            return bands

        # HANDLE LIST TO DICT CONVERSION
        if isinstance(bands, list):
            bands = {band: band for band in bands}

        output_bands = {}

        config = self.collection_mapping(collection)

        # Try to map each band
        for band in bands:
            if band in config[0]:
                output_bands[config[0][band]] = band
            # No band found with specified key (common band name)
            else:
                # Looking for band matching the specified value (asset name)
                matching_assets = [
                    key for key, value in config[0].items() if value == band
                ]

                if matching_assets:
                    output_bands[band] = band
        return output_bands
