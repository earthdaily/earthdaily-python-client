import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

project = "EarthDaily Python Client"
copyright = "2025, EarthDaily"
author = "EarthDaily"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "myst_parser",
    "sphinx_multiversion",
]

automodapi_writereprocessed = True
automodsumm_inherited_members = True
numpydoc_show_class_members = False

try:
    __version__ = get_version("earthdaily")
except PackageNotFoundError:
    __version__ = "0.0.0"

version = release = str(__version__)

sphinx_gallery_conf = {
    "backreferences_dir": os.path.join("_modules", "backreferences"),
    "doc_module": "earthdaily",
    "examples_dirs": "../examples",
    "filename_pattern": "",
    "ignore_pattern": "__",
    "gallery_dirs": ["_auto_examples"],
    "inspect_global_variables": False,
    "remove_config_comments": True,
    "plot_gallery": "True",
}

pygments_style = "sphinx"
nbsphinx_allow_errors = True
autosummary_generate = True
imported_members = True

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"

templates_path = ["templates"]
language = "en"
exclude_patterns = ["_build", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

smv_tag_whitelist = r"^1\.0\.0.*$"
smv_branch_whitelist = r"^(main|lts/0\.x)$"
smv_prefer_remote_refs = True
