from setuptools import find_packages, setup

# Retrieve release number from text file VERSION.
# See https://packaging.python.org/guides/single-sourcing-package-version/.
with open("earthdaily/__init__.py", encoding="utf8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].split('"')[1]

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="earthdaily",
    packages=find_packages(exclude=["tests"]),
    version=version,
    description="earthdaily: easy authentication, search and retrieval of Earth Data Store collections data",
    author="EarthDaily Agro",
    python_requires=">=3.10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "matplotlib<3.9",
        "joblib",
        "scipy",
        "psutil",
        "pystac-client>=0.7.0",
        "pystac",
        "odc-stac",
        "flox",
        "pandas>=2",
        "geopandas>=1",
        "requests",
        "xarray",
        "rasterio",
        "rioxarray",
        "tqdm",
        "python-dotenv",
        "rich",
        "dask<2025.3.0",
        "spyndex",
        "dask-image>=2024",
        "numba",
        "urllib3",
        "click",
        "toml",
    ],
    extras_require={
        "dev": [
            "ruff>=0.9.6,<1.0.0",
            "mypy>=1.0.0,<2.0.0",
        ]
    },
    include_package_data=True,
    package_data={"": ["*.geojson", "*.json"]},
    license="MIT",
    zip_safe=False,
    keywords=["Earth Data Store", "earthdaily", "earthdailyagro", "stac"],
    entry_points={
        "console_scripts": [
            "copy-earthdaily-credentials-template=earthdaily.utils.copy_credentials_template:cli"
        ],
    },
)
