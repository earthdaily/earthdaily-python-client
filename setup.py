from setuptools import setup, find_packages

# Retrieve release number from text file VERSION.
# See https://packaging.python.org/guides/single-sourcing-package-version/.
with open("earthdaily/__init__.py", encoding="utf8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].split('"')[1]
# with open('requirements.txt') as f:
#     required = f.read().splitlines()

setup(
    name="earthdaily",
    packages=[find_packages()],
    version=version,
    description="earthdaily: easy authentication, search and retrieval of Earth Data Store collections data",
    author="EarthDaily Agro",
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
    ],
    license="Commercial",
    zip_safe=False,
    keywords=["Earth Data Store", "earthdaily", "earthdailyagro", "stac"],
)
