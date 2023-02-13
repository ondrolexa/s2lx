# s2lx

[![main](https://github.com/ondrolexa/s2lx/actions/workflows/master.yml/badge.svg)](https://github.com/ondrolexa/s2lx/actions/workflows/master.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ondrolexa/s2lx/blob/master/LICENSE)

Simple Sentinel-2 tools

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

## Example

Download Sentinel-2 Level-2A or Level-1C product from Copernicus Open Access Hub in SAFE format and unzip. Now import `s2lx`:

    from s2lx import *

Open SAFE data:

    s = SAFE('/path/to/safename.SAFE/MTD_MSIL2A.xml')

You can preview whole scene:

    s.preview()

Clip region of interest (Note that projWin is defined in coordinate system of scene) and store in `S2` collection:

    d = s.clip(projWin=(440000, 5123000, 494000, 5093000), name='My Region')

To see the list of bands:

    d.bands

Individual bands could ba accessed as properties:

    d.b4.show()

You can use `Composite` class to create RGB composite:

    rgb = Composite(d.b4, d.b3, d.b2, name='True Color')
    rgb.show()

Bands and composites could be saved to GeoTIFF with `save` method:

    d.b4.save('b4.tif')
    rgb.save('truecolor.tif')

Bands support simple mathematical operations (addition, subtraction, division, multiplication)

    alt = Composite(d.b11/d.b12, d.b4/d.b2, d.b4/d.b11, name='Alterations')
    alt.show()

Bands could be filtered (check `s2lx.s2filters` for possible filters):

    b12f = d.b12.apply(median_filter(radius=4))

You can do PCA analysis using `S2.pca` method:

    p = d.pca()

To create RGB composite from first three principal components:

    pca = Composite(p.pc0, p.pc1, p.pc4, name='PCA')
    pca.show()

You can use also PCA to filter your data, i.,e. use only few PC to reconstruct dataset. Here we remove last four (from 9) components with lowest explained variance and reconstruct all bands:

    r = d.restored_pca(remove=(5,6,7,8))
    altr = Composite(r.b11/r.b12, r.b4/r.b2, r.b4/r.b11, name='Alterations filtered')
    altr.show()

## License

s2lx is free software: you can redistribute it and/or modify it under the terms of the MIT License. A copy of this license is provided in ``LICENSE`` file.
