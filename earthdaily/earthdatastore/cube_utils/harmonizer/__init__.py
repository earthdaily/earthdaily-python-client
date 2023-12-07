from datetime import datetime

import pytz
import xarray as xr
import numpy as np
import operator


class Harmonizer:
    def harmonize(items_collection, ds, cross_cal_items, assets):
        """
        Harmonize a dataset using cross_cal items from EarthDaily collection.

        Parameters
        ----------
        items_collection : TYPE
            DESCRIPTION.
        ds : TYPE
            DESCRIPTION.
        cross_cal_items : TYPE
            DESCRIPTION.
        assets : TYPE
            DESCRIPTION.

        Returns
        -------
        ds_ : TYPE
            DESCRIPTION.

        """
        if assets is None:
            assets = list(ds.data_vars.keys())

        scaled_dataset = {}

        # Initializing asset list
        for asset in assets:
            scaled_dataset[asset] = []

        # For each item in the datacube
        for idx, time in enumerate(ds.time.values):
            current_item = items_collection[idx]

            platform = current_item.properties["platform"]

            # Looking for platform/camera specific xcal coef
            platform_xcal_items = [
                item
                for item in cross_cal_items
                if item.properties["eda_cross_cal:source_platform"] == platform
                and Harmonizer.check_timerange(item, current_item.datetime)
            ]

            # at least one match
            matching_xcal_item = None
            if len(platform_xcal_items) > 0:
                matching_xcal_item = platform_xcal_items[0]
            else:
                # Looking for global xcal coef
                global_xcal_items = [
                    item
                    for item in cross_cal_items
                    if item.properties["eda_cross_cal:source_platform"] == ""
                    and Harmonizer.check_timerange(item, current_item.datetime)
                ]

                if len(global_xcal_items) > 0:
                    matching_xcal_item = cross_cal_items[0]

            if matching_xcal_item is not None:
                for ds_asset in assets:
                    # Loading Xcal coef for the specific band
                    bands_coefs = matching_xcal_item.properties["eda_cross_cal:bands"]

                    if ds_asset in bands_coefs:
                        asset_xcal_coef = matching_xcal_item.properties[
                            "eda_cross_cal:bands"
                        ][ds_asset]
                        # By default, we take the first item we have
                        scaled_asset = Harmonizer.apply_to_asset(
                            asset_xcal_coef[0][ds_asset],
                            ds[ds_asset].loc[dict(time=time)],
                            ds_asset,
                        )
                        scaled_dataset[ds_asset].append(scaled_asset)
                    else:
                        scaled_dataset[ds_asset].append(
                            ds[ds_asset].loc[dict(time=time)]
                        )

        ds_ = []
        for k, v in scaled_dataset.items():
            ds_k = []
            for d in v:
                ds_k.append(d)
            ds_.append(xr.concat(ds_k, dim="time"))
        ds_ = xr.merge(ds_).sortby("time")
        ds_.attrs = ds.attrs

        return ds_

    def xcal_functions_parser(functions, dataarray: xr.DataArray):
        xscaled_dataarray = []
        for idx_function, function in enumerate(functions):
            coef_range = [function["range_start"], function["range_end"]]
            for idx_coef, coef_border in enumerate(coef_range):
                for single_operator, threshold in coef_border.items():
                    xr_condition = getattr(operator, single_operator)(
                        dataarray, threshold
                    )
                    if idx_coef == 0:
                        ops = xr_condition
                    else:
                        ops = np.logical_and(ops, xr_condition)
            xscaled_dataarray.append(
                dict(condition=ops, scale=function["scale"], offset=function["offset"])
            )

        for op in xscaled_dataarray:
            dataarray = xr.where(
                op["condition"], dataarray * op["scale"] + op["offset"], dataarray
            )
        return dataarray

    def apply_to_asset(functions, dataarray: xr.DataArray, band_name):
        if len(functions) == 1:
            # Single function
            dataarray = dataarray * functions[0]["scale"] + functions[0]["offset"]
        else:
            # Multiple functions
            # TO DO : Replace x variable and the eval(xr_where_string) by a native function
            dataarray = Harmonizer.xcal_functions_parser(functions, dataarray)
        return dataarray

    def check_timerange(xcal_item, item_datetime):
        start_date = datetime.strptime(
            xcal_item.properties["published"], "%Y-%m-%dT%H:%M:%SZ"
        )
        start_date = start_date.replace(tzinfo=pytz.UTC)

        if not isinstance(item_datetime, datetime):
            item_datetime = datetime.strptime(item_datetime, "%Y-%m-%dT%H:%M:%SZ")
        item_datetime = item_datetime.replace(tzinfo=pytz.UTC)

        if "expires" in xcal_item.properties:
            end_date = datetime.strptime(
                xcal_item.properties["expires"], "%Y-%m-%dT%H:%M:%SZ"
            )
            end_date = end_date.replace(tzinfo=pytz.UTC)

            return start_date <= item_datetime <= end_date
        else:
            return start_date <= item_datetime
