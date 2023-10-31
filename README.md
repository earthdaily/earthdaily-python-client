# Earthdaily Python Package
[![PyPI version](https://badge.fury.io/py/earthdaily.png)](https://badge.fury.io/py/earthdaily)
[![Documentation](https://img.shields.io/badge/Documentation-html-green.svg)](https://geosys.github.io/earthdaily-python-client/)
[![pytest-main](https://github.com/GEOSYS/earthdaily-python-client/actions/workflows/pytest-prod.yaml/badge.svg)](https://github.com/GEOSYS/earthdaily-python-client/actions/workflows/pytest-prod.yaml)

## Your Gateway to the Stac Catalog Earth Data Store

In the realm of geospatial data analysis and Earth observation, the EarthDaily Python package emerges as a powerful toolset that seamlessly connects you to the vast and invaluable Stac catalog Earth Data Store. This package is designed with the vision of simplifying and optimizing your workflow, ensuring that you can harness the full potential of Earth observation data with ease and efficiency.

Our package is built upon a foundation of best practices, meticulously crafted to elevate your data analysis experience. With EarthDaily, you can effortlessly navigate the complexities of datacube creation, including crucial processes like conversion to reflectance and automatic clipping to your area of interest. Additionally, we've taken care to make EarthDaily fully compatible with Dask, enabling you to scale your data preprocessing tasks with confidence and precision.

## Install

### Using pip 

`pip install earthdaily`

### Planned : Using conda/mamba

## Authentication
Authentication credentials are accessed from environment variables. As a convenience python-dotenv is supported. 
Copy the `.env.sample` file and rename to simply `.env` and update with your credentials. This file is gitignored. 
Then add to your script/notebook:

```python3
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
```