<div id="top"></div>
<!-- PROJECT SHIELDS -->
<!--
*** See the bottom of this document for the declaration of the reference variables
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<p>
  <h1 >EarthDaily Python Client</h3>

  <p>
    Your gateway to the Earth Data Store STAC Catalog.
    <br />
    <a href="https://earthdailyagro.com/"><strong>Who we are</strong></a>
    <br />
    <br />
    <a href="https://github.com/earthdaily/earthdaily-python-client/">Project description</a>
    ·
    <a href="https://github.com/earthdaily/earthdaily-python-client/issues">Report Bug</a>
    ·
    <a href="https://github.com/earthdaily/earthdaily-python-client/issues">Request Feature</a>
  </p>
</p>


<div>

[![PyPI version](https://badge.fury.io/py/earthdaily.png)](https://badge.fury.io/py/earthdaily)
[![Documentation](https://img.shields.io/badge/Documentation-html-green.svg)](https://earthdaily.github.io/earthdaily-python-client/)
[![pytest-main](https://github.com/earthdaily/earthdaily-python-client/actions/workflows/pytest-prod.yaml/badge.svg)](https://github.com/earthdaily/earthdaily-python-client/actions/workflows/pytest-prod.yaml)

</div>


<!--[![Stargazers][GitStars-shield]][GitStars-url]-->
<!--[![Forks][forks-shield]][forks-url]-->
<!--[![Stargazers][stars-shield]][stars-url]-->


<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#support-development">Support development</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#copyrights">Copyrights</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

In the realm of geospatial data analysis and Earth observation, the EarthDaily Python package emerges as a powerful toolset that seamlessly connects you to the vast and invaluable Stac catalog Earth Data Store. This package is designed with the vision of simplifying and optimizing your workflow, ensuring that you can harness the full potential of Earth observation data with ease and efficiency.

Our package is built upon a foundation of best practices, meticulously crafted to elevate your data analysis experience. With EarthDaily, you can effortlessly navigate the complexities of datacube creation, including crucial processes like conversion to reflectance and automatic clipping to your area of interest. Additionally, we've taken care to make EarthDaily fully compatible with Dask, enabling you to scale your data preprocessing tasks with confidence and precision.


## Features

See [documentation](https://earthdaily.github.io/earthdaily-python-client/) for more information

## Getting started

### Prerequisites

Make sure you have valid EDS credentials. If you need to get trial access, please contact us.

This package has been tested on Python 3.10, 3.11 and 3.12.


### Installation

#### From pypi, using pip

`pip install earthdaily`

#### From repository, using mamba, conda and pip

```console
# Clone the repository and go inside
git clone git@github.com:earthdaily/earthdaily-python-client.git
cd earthdaily-python-client

# Create a virtual environment named earthdaily and install package dependencies
mamba env create -n earthdaily -f requirements.yml
conda activate earthdaily

# Install package in editable mode
pip install -e .
```

### Authentication

In order to generate a credentials file with the right variables and layout, you can use the `copy-earthdaily-credentials-template` script that is installed along the earthdaily package. It can generate credentials file as JSON, TOML or .env.
Generate the credentials file of your liking and edit it to insert your credentials.


#### From the default credentials file

Create a TOML credentials file in the default location:
* "$HOME/.earthdaily/credentials" on linux
* "$USERPROFILE/.earthdaily/credentials" on Windows

```console
copy-earthdaily-credentials-template --default
```

Edit it to insert your credentials.
The following code will automatically find and use the credentials for authentification.

```python
from earthdaily import EarthDataStore
eds = EarthDataStore()
```

#### From a JSON file

Authentication credentials can be given as an input JSON file.
You can generate a JSON credentials file with the following command:

```console
copy-earthdaily-credentials-template --json "/path/to/the/credentials_file.json"
```

Edit it to insert your credentials.
Then use it as input for authentification:

```python
from pathlib import Path
from earthdaily import EarthDataStore
eds = EarthDataStore(json_path = Path("/path/to/the/credentials_file.json"))
```

#### From a TOML file

Authentication credentials can be given as input with a TOML file.
You can generate a TOML credentials file with the following command:

```console
copy-earthdaily-credentials-template --toml "/path/to/the/credentials_file"
```

Edit it to insert your credentials.
Then use it as input for authentification:

```python
from pathlib import Path
from earthdaily import EarthDataStore
eds = EarthDataStore(toml_path = Path("/path/to/the/credentials_file"))
```

#### From environment variables

Authentication credentials can be automatically parsed from environment variables.
As a convenience python-dotenv is supported. 

You can generate a .env credentials file with the following command:

```console
copy-earthdaily-credentials-template --env "/path/to/the/.env"
```

Or you can copy the `.env.sample` file from the repository and rename it to `.env`.
Edit it to insert your credentials.
Then add to your script/notebook:

```python
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
```

### Usage

See the documentation for more information.

### Support development

If you find this package useful, please consider supporting its development.

<!-- CONTRIBUTING -->
## Support development

If this project has been useful, that it helped you or your business to save precious time, don't hesitate to give it a star.

<p align="right">(<a href="#top">back to top</a>)</p>

## License

Distributed under the MIT License. 

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

For any additonal information, please [email us](mailto:sales@earthdailyagro.com).

<p align="right">(<a href="#top">back to top</a>)</p>

## Copyrights

© EarthDaily | All Rights Reserved.

<p align="right">(<a href="#top">back to top</a>)</p>