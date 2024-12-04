[![PyPI version](https://badge.fury.io/py/earthdaily.svg)](https://badge.fury.io/py/earthdaily)
[![Documentation](https://img.shields.io/badge/Documentation-Online-brightgreen.svg)](https://earthdaily.github.io/earthdaily-python-client/)
[![Build Status](https://github.com/earthdaily/earthdaily-python-client/actions/workflows/pytest-prod.yaml/badge.svg)](https://github.com/earthdaily/earthdaily-python-client/actions)
![Python Versions](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue)
[![Issues](https://img.shields.io/github/issues/earthdaily/earthdaily-python-client.svg)](https://github.com/earthdaily/earthdaily-python-client/issues)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

# EarthDaily Python Client

Your gateway to the Earth Data Store STAC Catalog.

[EarthDaily Homepage](https://earthdaily.com) |
[Report Bug](https://github.com/earthdaily/earthdaily-python-client/issues) |
[Request Feature](https://github.com/earthdaily/earthdaily-python-client/issues)

---

The **EarthDaily Python client** simplifies access to the Earth Data Store STAC catalog and streamlines workflows for geospatial data analysis and Earth observation. It automates key preprocessing tasks, making it easier to work with Earth observation data.

## Features

- **Easy Data Access**: Connect directly to the Earth Data Store STAC catalog.
- **Automated Datacube Creation**: Includes reflectance conversion and clipping to areas of interest.
- **Scalable Processing**: Fully compatible with Dask for handling large datasets.

This package is designed to make geospatial workflows more efficient and accessible for researchers and analysts working with Earth observation data.

### Prerequisites

Make sure you have valid Earth Data Store authentication credentials. To request access, please [contact us](mailto:sales@earthdailyagro.com).


## Installation

### From PyPI, using pip

`pip install earthdaily`

### From Repository using Conda and pip

```console
# Clone the repository and navigate inside
git clone git@github.com:earthdaily/earthdaily-python-client.git
cd earthdaily-python-client

# Create a conda environment and install dependencies
conda env create -n earthdaily -f requirements.yml
conda init
conda activate earthdaily

# Install package in editable mode
pip install -e .
```

### Authentication from Environment Variables

Authentication credentials can be automatically parsed from environment variables.
The [python-dotenv](https://github.com/theskumar/python-dotenv) package is supported for convenience.

Rename the `.env.sample` file in this repository to `.env` and enter your credentials.

In your script or notebook, add:

```python
from dotenv import load_dotenv

load_dotenv(".env")  # Load environment variables from .env file
```

## Support development

If this project has been helpful and saved you time, please consider giving it a star ⭐


## Contact

For additional information, please [email us](mailto:sales@earthdailyagro.com).


## Copyrights

© EarthDaily | All Rights Reserved.
