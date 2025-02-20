# Configuration file for the Sphinx documentation builder.

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
project = "earthdaily"
copyright = "2024, EarthDailyAgro"
author = "Geosys/EarthDailyAgro"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "myst_parser",
]

automodapi_writereprocessed = True
automodsumm_inherited_members = True
numpydoc_show_class_members = False
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))


try:
    __version__ = get_version("earthdaily")
except PackageNotFoundError:
    __version__ = "0.0.0"

source_suffix = [".rst"]
version = str(__version__)
release = str(__version__)

sphinx_gallery_conf = {
    "backreferences_dir": os.path.join("_modules", "backreferences"),
    "doc_module": "earthdaily",
    # path to your examples scripts
    "examples_dirs": "../examples",
    "filename_pattern": "",
    # path where to save gallery generated examples
    "ignore_pattern": "__",
    "gallery_dirs": ["_auto_examples"],
    # avoid generating too many cross links
    "inspect_global_variables": False,
    "remove_config_comments": True,
    "plot_gallery": "True",
}
pygments_style = "sphinx"
nbsphinx_allow_errors = True
autosummary_generate = True

imported_members = True

autoclass_content = "both"

templates_path = ["templates"]

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

source_suffix = [".rst", ".md"]

pdf_documents = [
    ("index", "earthdaily_documentation", "earthdaily doc", "EarthDaily Agro"),
]

todo_include_todos = False
