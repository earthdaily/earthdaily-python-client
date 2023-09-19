# earthdaily python package

The earthdaily python package provides an API to use the Stac catalog Earth Data Store. Based onto 
It uses best practices for datacube creation (convert to reflectance, automatic clipping to area of interest...), dask compatible preprocesses...

## Install

### Using mamba (recommended)

We recommend using mamba instead of conda.
From a fresh install, download latest mambaforge : 
https://github.com/conda-forge/miniforge#mambaforge

Then in your terminal (unix) or in Mambaforge Prompt on Windows install in your env (here we name it earthdaily. If the env does not exist, it will create it):
> mamba env update --name earthdaily --file requirements.yml