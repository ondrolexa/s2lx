=====
Usage
=====

Start by downloading Sentinel-2 Level-2A or Level-1C product from Copernicus Open Access Hub in SAFE format and unzip. Now import `s2lx`:

.. code-block:: python

    from s2lx import *

Open SAFE data:

.. code-block:: python

    s = SAFE('/path/to/safename.SAFE/MTD_MSIL2A.xml')

You can preview whole scene:

.. code-block:: python

    s.preview()

Clip region of interest (Note that projWin is defined in coordinate system of scene) and store in `S2` collection:

.. code-block:: python

    projWin = (440000, 5123000, 494000, 5093000)
    d = s.clip(projWin, name='My Region')

To see the list of bands:

.. code-block:: python

    d.bands

Individual bands could ba accessed as properties:

.. code-block:: python

    d.b4.show()

You can use `Composite` class to create RGB composite:

.. code-block:: python

    rgb = Composite(d.b4, d.b3, d.b2, name='True Color')
    rgb.show()

Bands and composites could be saved to GeoTIFF with `save` method:

.. code-block:: python

    d.b4.save('b4.tif')
    rgb.save('truecolor.tif')

Bands support simple mathematical operations (addition, subtraction, division, multiplication)

.. code-block:: python

    alt = Composite(d.b11/d.b12, d.b4/d.b2, d.b4/d.b11, name='Alterations')
    alt.show()

Bands could be filtered (check `s2lx.s2filters` for possible filters):

.. code-block:: python

    b12f = d.b12.apply(median_filter(radius=4))

You can do PCA analysis using `S2.pca` method:

.. code-block:: python

    p = d.pca()

To create RGB composite from first three principal components:

.. code-block:: python

    pca = Composite(p.pc0, p.pc1, p.pc4, name='PCA')
    pca.show()

You can use also PCA to filter your data, i.,e. use only few PC to reconstruct dataset. Here we remove last four (from 9) components with lowest explained variance and reconstruct all bands:

.. code-block:: python

    r = d.restored_pca(remove=(5,6,7,8))
    altr = Composite(r.b11/r.b12, r.b4/r.b2, r.b4/r.b11, name='Alterations filtered')
    altr.show()

