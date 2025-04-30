# s2lx - simple Sentinel-2 tools

[![main](https://github.com/ondrolexa/s2lx/actions/workflows/testing.yml/badge.svg)](https://github.com/ondrolexa/s2lx/actions/workflows/testing.yml)
[![Documentation](https://readthedocs.org/projects/apsg/badge/?version=latest)](https://apsg.readthedocs.io/en/stable/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ondrolexa/s2lx/blob/master/LICENSE)

## What is s2lx

s2lx is a python toolbox to simplify explorative work with Sentinel-2 multispectral imagery. It allows quickly extract data from region of interest,
easily manipulate a test processing and visualiztion strategies. In addition, it could be used for quick merging and reprojecting of data.

## How to install

Easiest way to install **s2lx** is to use conda package management system. Create conda environment from the included `environment.yml` file:

    conda env create -f environment.yml

Then activate the new environment:

    conda activate s2lx

and install s2lx using pip:

    pip install https://github.com/ondrolexa/s2lx/archive/master.zip

### Upgrade existing installation

To upgrade an already installed **s2lx** to the latest release:

    pip install --upgrade https://github.com/ondrolexa/s2lx/archive/master.zip

## Documentation

Check [documentation](https://s2lx.readthedocs.io/en/latest/) for more details and examples.

## License

s2lx is free software: you can redistribute it and/or modify it under the terms of the MIT License. A copy of this license is provided in ``LICENSE`` file.
