from setuptools import setup, find_packages

# Retrieve release number from text file VERSION.
# See https://packaging.python.org/guides/single-sourcing-package-version/.
with open("earthdaily/__init__.py", encoding="utf8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].split('"')[1]

setup(
    name="earthdaily",
    packages=find_packages(),
    version=version,
    description="earthdaily: easy authentication, search and retrieval of Earth Data Store collections data",
    author="EarthDaily Agro",
    python_requires=">=3.10",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "joblib",
        "psutil",
        "xarray",
        "pandas",
        "geopandas",
        "rasterio",
        "pystac-client",
        "requests",
        "xarray",
        "rioxarray",
        "h5netcdf ",
        "netcdf4",
        "pystac",
        "stackstac",
        "odc-stac",
        "tqdm",
        "python-dotenv",
        "rich",
        "python-dotenv",
    ],
    license="MIT",
    zip_safe=False,
    keywords=["Earth Data Store", "earthdaily", "earthdailyagro", "stac"],
)
