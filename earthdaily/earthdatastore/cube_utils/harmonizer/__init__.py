from datetime import datetime

import pytz
import xarray as xr


class Harmonizer:
    def harmonize(self, items_collection, ds, cross_cal_items, assets):
        if assets is None:
            assets = list(ds.data_vars.keys())

        scaled_dataset = {}

        # Initializing asset list
        for asset in assets:
            scaled_dataset[asset] = []

        # For each item in the datacube
        for idx, time in enumerate(ds.time.values):
            current_item = items_collection[idx]
            print("****")
            print(current_item.id)

            platform = current_item.properties["platform"]

            # Looking for platform/camera specific xcal coef
            platform_xcal_items = [
                item
                for item in cross_cal_items
                if item.properties["eda_cross_cal:source_platform"] == platform
                and self.check_timerange(item, current_item.datetime)
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
                    and self.check_timerange(item, current_item.datetime)
                ]

                if len(global_xcal_items) > 0:
                    matching_xcal_item = cross_cal_items[0]

            if matching_xcal_item is not None:
                print(matching_xcal_item.id)
                for ds_asset in assets:
                    # Loading Xcal coef for the specific band
                    bands_coefs = matching_xcal_item.properties["eda_cross_cal:bands"]

                    if ds_asset in bands_coefs:
                        asset_xcal_coef = matching_xcal_item.properties[
                            "eda_cross_cal:bands"
                        ][ds_asset]
                        # By default, we take the first item we have
                        scaled_asset = self.apply_to_asset(
                            asset_xcal_coef[0][ds_asset],
                            ds[[ds_asset]].loc[dict(time=time)],
                            ds_asset,
                        )
                        scaled_dataset[ds_asset].append(scaled_asset)
                    else:
                        scaled_dataset[ds_asset].append(
                            ds[[ds_asset]].loc[dict(time=time)]
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

    def xcal_functions_parser(self, functions):
        possible_range = ["ge", "gt", "le", "lt"]

        operator_mapping = {
            "lt": "<",
            "le": "<=",
            "ge": ">=",
            "gt": ">",
        }

        xarray_where_concat = ""
        for idx_function, function in enumerate(functions):
            coef_range = [function["range_start"], function["range_end"]]

            for idx_coef, coef_border in enumerate(coef_range):
                for range in possible_range:
                    if range in coef_border:
                        threshold = coef_border[range]
                        condition = operator_mapping.get(range)
                        if idx_coef == 0:
                            xarray_where_concat += f"xr.where((x{condition}{threshold})"
                        else:
                            xarray_where_concat += f" & (x{condition}{threshold})"

            xarray_where_concat += f',x * {function["scale"]} + {function["offset"]},'

            if idx_function == len(functions) - 1:
                xarray_where_concat += "x"
                function_parenthesis = 0

                while function_parenthesis < len(functions):
                    xarray_where_concat += ")"
                    function_parenthesis += 1

        return xarray_where_concat

    def apply_to_asset(self, functions, asset, band_name):
        if len(functions) == 1:
            # Single function
            return asset * functions[0]["scale"] + functions[0]["offset"]
        else:
            # Multiple functions
            # TO DO : Replace x variable and the eval(xr_where_string) by a native function
            x = asset[band_name]
            xr_where_string = self.xcal_functions_parser(functions)
            asset[band_name] = eval(xr_where_string)
            return asset

    def check_timerange(self, xcal_item, item_datetime):
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
