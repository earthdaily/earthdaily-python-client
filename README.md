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

Make sure you have valid Earth Data Store authentication credentials. These can be retrieved from the [EarthDaily Account Management](https://console.earthdaily.com/account) page.

<p align="center">
<img src="https://github.com/earthdaily/earthdaily-python-client/raw/main/docs/assets/images/account.png" width="450">
</p>

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

Rename the `.env.sample` file in this repository to `.env` and enter your Earth Data Store authentication credentials. 
Note this file is gitingored and will not be committed.

In your script or notebook, add:

```python
from dotenv import load_dotenv

load_dotenv(".env")  # Load environment variables from .env file
```

## Quickstart
To help you get started quickly, we provide a `quickstart.ipynb` Jupyter notebook that demonstrates how to use the EarthDaily Python client. You only need your `.env` file with your authentication credentials to run it. 

Simply open the notebook, load your environment variables as shown above, and follow the instructions to begin accessing and analyzing Earth observation data.

## Support development

If this project has been helpful and saved you time, please consider giving it a star ⭐

## Copyrights

© EarthDaily | All Rights Reserved.
