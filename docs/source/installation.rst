============
Installation
============

Easiest way to install **s2lx** is to use conda (or mamba) package management system. For convinience you can create separate conda environment using included `environment.yml` file::

    $ conda env create -f environment.yml

Then activate the new environment::

    $ conda activate s2lx

and install s2lx using pip::

    (s2lx)$ pip install https://github.com/ondrolexa/s2lx/archive/master.zip

To upgrade an already installed **s2lx** to the latest release::

    (s2lx)$ pip install --upgrade https://github.com/ondrolexa/s2lx/archive/master.zip

In any other scenario, the `s2lx` requires following dependencies:

 * `python>=3.9`
 * `numpy`
 * `matplotlib`
 * `scipy`
 * `gdal`
 * `shapely>=2`
 * `pyproj`
 * `scikit-image`
 * `scikit-learn`