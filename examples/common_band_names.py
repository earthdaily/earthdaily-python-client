"""
EarthDaily unique common band names
=================================================================

For a better interoperability between sensors."""

##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily.earthdatastore.cube_utils import asset_mapper
from rich.table import Table
from rich.console import Console

console = Console(force_interactive=True)

##############################################################################
# Show each collection with their earthdaily common band names
# --------------------------------------------------------------
# For band names where several bands are available (rededge) it has been chosen
# to use the central wavelength (rededge70 is rededge1 of sentinel-2 for example).
#

for collection, assets in asset_mapper._asset_mapper_config.items():
    table = Table(
        "asset",
        "EarthDaily Common band name",
        title=f"Earthdaily common names for {collection}",
    )
    for common_name, asset in assets[0].items():
        table.add_row(asset, common_name)
    console.print(table)
