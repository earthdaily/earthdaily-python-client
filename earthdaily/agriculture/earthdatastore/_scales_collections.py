scale_factor_collections = {
    "landsat-c2l2-sr": [
        dict(
            assets=[
                "red",
                "blue",
                "green",
                "nir",
                "nir08",
                "swir16",
                "swir22",
                "coastal",
            ],
            scale=0.0000275,
            offset=-0.2,
            nodata=0,
        )
    ],
    "landsat-c2l2-st": [
        dict(
            assets=["lwir", "lwir11", "lwir12"],
            scale=0.00341802,
            offset=149.0,
            nodata=0,
        )
    ],
    "landsat-c2l1": [
        dict(
            assets=[
                "red",
                "blue",
                "green",
                "nir",
                "nir08",
                "swir16",
                "swir22",
                "coastal",
            ],
            scale=0.0001,
            offset=0,
            nodata=-9999,
        ),
        dict(
            assets=[
                "lwir",
                "lwir11",
                "lwir12",
            ],
            scale=0.1,
            offset=0,
            nodata=-9999,
        ),
    ],
}
