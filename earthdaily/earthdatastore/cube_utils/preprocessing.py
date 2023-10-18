from rasterio import features


def rasterize(gdf, datacube, all_touched=True):
    gdf["geometry"] = gdf.to_crs(datacube.rio.crs).clip_by_rect(*datacube.rio.bounds())
    shapes = ((gdf.iloc[i].geometry, i + 1) for i in range(gdf.shape[0]))

    # rasterize features to use numpy/scipy to avoid polygon clipping
    feats = features.rasterize(
        shapes=shapes,
        fill=0,
        out_shape=datacube.rio.shape,
        transform=datacube.rio.transform(),
        all_touched=all_touched,
    )
    return feats
