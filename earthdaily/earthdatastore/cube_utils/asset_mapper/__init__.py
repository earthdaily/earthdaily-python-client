from . import _asset_mapper_config


class AssetMapper:
    def map_collection_bands(self, collection, bands):
        if isinstance(bands,dict):
            return bands
        if collection not in _asset_mapper_config.asset_mapper_collections:
            return bands

        # HANDLE LIST TO DICT CONVERSION
        if isinstance(bands, list):
            bands = {band: band for band in bands}

        output_bands = {}

        config = _asset_mapper_config.asset_mapper_collections[collection]

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