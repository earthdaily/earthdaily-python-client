"""
Explore collection metadata using earthdaily
===============================================================

"""
##############################################################################
# Import librairies
# -------------------------------------------

from earthdaily import earthdatastore
from rich.table import Table
from rich.console import Console

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

console = Console()


##############################################################################
# Init earthdaily with earthdatastore
# -------------------------------------------------

eds = earthdatastore.Auth()

##############################################################################
# Explore available collections
# -------------------------------------------

table = Table("Available collections")
for t in eds.explore():
    table.add_row(t)
console.print(table)


##############################################################################
# Explore a specific collection
# -------------------------------------------

collection = eds.explore("venus-l2a")
console.log(collection.properties)

##############################################################################
# List properties available per item
# -------------------------------------------

table = Table("properties", "values", "dtype", title=f"Properties for {collection}")
for k, v in collection.item_properties.items():  # item_properties is a dict
    table.add_row(k, str(v), type(v).__name__)
console.print(table)

##############################################################################
# Read assets and metadata
# -------------------------------------------

table = Table("assets", "common_name", "description", title=f"Assets for {collection}")
for asset in collection.assets():
    table.add_row(
        asset,
        collection.assets(asset).get("eo:bands", [{}])[0].get("common_name"),
        collection.assets(asset).get("eo:bands", [{}])[0].get("description"),
    )
console.print(table)
