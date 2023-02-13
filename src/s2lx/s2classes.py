# -*- coding: utf-8 -*-

import json
import numbers
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib import colors, path
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from osgeo import gdal, gdal_array
from shapely import geometry, ops, wkt
from sklearn import decomposition
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler


class SAFE:
    """Class to manipulate Sentinel-2 product

    The SAFE format has been designed to act as a common format for
    archiving and conveying data within ESA Earth Observation archiving
    facilities. The SAFE format wraps a folder containing image data in
    a binary data format and product metadata in XML.

    The Level-2A prototype product is an orthorectified product providing
    Bottom-Of-Atmosphere (BOA) reflectances, and basic pixel classification
    (including classes for different types of cloud). The generation of
    this prototype product is carried out by the User from Level-1C products.

    The Level-2A image data product uses the same tiling, encoding
    and filing structure as Level-1C.

    Attributes:
        datasets (dict): Dictionary containing information abou available sub-
            datasets. Commonly "10m", "20m", "60m" and "TCI"
        meta (dict): Dictionary with SAFE format metadata

    """

    def __init__(self, xml):
        """Open SAFE format dataset

        The Sentinel-2 Level-2A and Level-1C are supported. The data could be
        downloaded from The Copernicus Open Access Hub.

        Args:
            xml (str): path to MTD_MSIL2A.xml file in root of SAFE folder.

        """
        xml = Path(xml)
        if xml.is_file():
            ds = gdal.Open(str(xml), gdal.GA_ReadOnly)
            self.datasets = {}
            self.meta = ds.GetMetadata()
            for xmlfile, desc in ds.GetSubDatasets():
                dsname, crs = xmlfile.split(":")[2:]
                crs = crs.replace("_", ":")
                sds = gdal.Open(xmlfile, gdal.GA_ReadOnly)
                meta = sds.GetMetadata()
                bands = {}
                for n in range(sds.RasterCount):
                    rb = sds.GetRasterBand(n + 1)
                    mtd = rb.GetMetadata()
                    name = mtd["BANDNAME"]
                    del mtd["BANDNAME"]
                    mtd["n"] = n + 1
                    bands[name] = mtd
                self.datasets[dsname] = dict(
                    xml=xmlfile,
                    crs=crs,
                    desc=desc,
                    bands=bands,
                    meta=meta,
                    transform=sds.GetGeoTransform(),
                    projection=sds.GetProjection(),
                )
                sds = None
            ds = None
        else:
            print("Error: No valid filename")

    def __repr__(self):
        txt = f'{self.meta["DATATAKE_1_SPACECRAFT_NAME"]} '
        txt += f'{self.meta["PRODUCT_TYPE"]} '
        txt += f'{self.meta["PROCESSING_LEVEL"]}\n'
        txt += f'Sensing date {self.meta["DATATAKE_1_DATATAKE_SENSING_START"]}\n'
        for d in self.datasets:
            txt += f'Dataset {d} bands: {sorted(list(self.datasets[d]["bands"].keys()))}\n'
        return txt

    def preview(self, **kwargs):
        """Show the scene TCI image

        Keyword Args:
            figsize (tuple): matplotlib Figure size

        """
        sds = gdal.Open(self.datasets["TCI"]["xml"])
        tci = np.dstack(
            (
                sds.GetRasterBand(self.datasets["TCI"]["bands"]["B4"]["n"]).ReadAsArray(),
                sds.GetRasterBand(self.datasets["TCI"]["bands"]["B3"]["n"]).ReadAsArray(),
                sds.GetRasterBand(self.datasets["TCI"]["bands"]["B2"]["n"]).ReadAsArray(),
            )
        )
        figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        # plot
        f, ax = plt.subplots(figsize=figsize)
        ax.imshow(tci)
        ax.set_title(self.meta["PRODUCT_URI"])
        ax.set_aspect(1)
        f.tight_layout()
        plt.show()

    @property
    def crs(self):
        """pyproj.CRS: coordinate reference system of scene"""
        return pyproj.CRS(self.datasets["20m"]["crs"])

    def get(self, dataset, band):
        """Get band as numpy array

        Args:
            dataset (str): matplotlib Figure size
            band (str): matplotlib Figure size

        Returns:
            Tuple of band numpy array and metadata dict

        """
        if dataset in self.datasets:
            if band in self.datasets[dataset]["bands"]:
                sds = gdal.Open(self.datasets[dataset]["xml"])
                meta = self.datasets[dataset]["bands"][band]
                return sds.GetRasterBand(meta["n"]).ReadAsArray(), meta

    def clip_geojson(self, geojson, res=20, name="Clip", include8a=False):
        """Clip scene by polygon stored in GeoJSON file

        Clip all bands in scene by rectangle defined as bounds of polygon
        stored in GeoJSON file. All bands in resulting collection have same
        resolution. For res=20, the '10m' dataset bands are downsampled,
        while for res=10, the bands of '20m' dataset are upsampled.

        Note:
            This method assume that CRS of GeoJSON and scene is identical

        Args:
            geojson (str): filename of GeoJSON with polygon feature

        Keyword Args:
            res (int): resolution of clipped bands. Default 20
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Dafault False

        Returns:
            `s2lx.S2` collection

        """
        with open(geojson) as f:
            features = json.load(f)["features"]
        bnds = geometry.shape(features[0]["geometry"]).buffer(0).bounds
        return self.clip(
            (bnds[0], bnds[3], bnds[2], bnds[1]),
            res=res,
            name=name,
            include8a=include8a,
        )

    def clip(self, projWin, res=20, name="Clip", include8a=False):
        """Clip scene by coordinates

        Clip all bands in scene by rectangle defined by subwindow. All bands
        in resulting collection have same resolution. For res=20, the '10m'
        dataset bands are downsampled, while for res=10, the bands of '20m'
        dataset are upsampled.

        Note:
            This method assume that coordinates of subwindow are given in
            CRS of scene

        Args:
            projWin (tuple): subwindow to clip (ulx, uly, lrx, lry)

        Keyword Args:
            res (int): resolution of clipped bands. Default 20
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Dafault False

        Returns:
            `s2lx.S2` collection

        """
        if projWin is not None:
            wgs84 = pyproj.CRS("EPSG:4326")
            reproject = pyproj.Transformer.from_crs(wgs84, self.crs, always_xy=True).transform
            # safe footprint
            footprint = ops.transform(reproject, wkt.loads(self.meta["FOOTPRINT"])).buffer(-200)
            # prepare mask
            cg = np.arange(projWin[0] + res // 2, projWin[2], res)
            rg = np.arange(projWin[1] - res // 2, projWin[3], -res)
            xx, yy = np.meshgrid(cg, rg)
            footpath = path.Path(np.array(footprint.exterior.xy).T)
            mask = np.invert(
                np.reshape(
                    footpath.contains_points(np.array([xx.ravel(), yy.ravel()]).T),
                    xx.shape,
                )
            )
            # go
            meta = self.meta.copy()
            rasters = []
            # 20m
            sds = gdal.Translate(
                "/vsimem/in_memory_output.tif",
                gdal.Open(self.datasets["20m"]["xml"]),
                projWin=projWin,
                xRes=res,
                yRes=res,
                resampleAlg=gdal.GRA_Average,
            )
            for band in ["B5", "B6", "B7", "B8A", "B11", "B12"]:
                bmeta = self.datasets["20m"]["bands"][band].copy()
                rb = sds.GetRasterBand(bmeta.pop("n"))
                vals = rb.ReadAsArray().astype(np.int16)
                if band == "B8A":
                    if include8a:
                        rasters.append(
                            Band(
                                "a8",
                                np.ma.masked_array(vals, mask),
                                sds.GetGeoTransform(),
                                sds.GetProjection(),
                            )
                        )
                        meta["a8"] = bmeta
                else:
                    rasters.append(
                        Band(
                            band.lower(),
                            np.ma.masked_array(vals, mask),
                            sds.GetGeoTransform(),
                            sds.GetProjection(),
                        )
                    )
                    meta[band.lower()] = bmeta
            sds = gdal.Translate(
                "/vsimem/in_memory_output.tif",
                gdal.Open(self.datasets["10m"]["xml"]),
                projWin=projWin,
                xRes=res,
                yRes=res,
                resampleAlg=gdal.GRA_Average,
            )
            for band in ["B4", "B3", "B2", "B8"]:
                bmeta = self.datasets["10m"]["bands"][band].copy()
                rb = sds.GetRasterBand(bmeta.pop("n"))
                vals = rb.ReadAsArray().astype(np.int16)
                rasters.append(
                    Band(
                        band.lower(),
                        np.ma.masked_array(vals, mask),
                        sds.GetGeoTransform(),
                        sds.GetProjection(),
                    )
                )
                meta[band.lower()] = bmeta
            return S2(*rasters, meta=meta, name=name)

    def gclip(self, name="Clip", include8a=False, band="B12"):
        """Quick clip by rectangular selection

        Clip all bands in scene by rectangular selection drawn by mouse.
        All bands in resulting collection have 20m resolution and '10m'
        dataset bands are downsampled.

        Note:
            Draw and modify selection by mouse. Clip by keypress enter.

        Keyword Args:
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Dafault False

        Returns:
            `s2lx.S2` collection

        """

        def onselect_function(eclick, erelease):
            pass

        def on_press(event):
            sys.stdout.flush()
            if event.key == "enter":
                rect_selector.set_active(False)
                plt.close(f)

        f, ax = plt.subplots()
        dt = self.get("20m", band)[0]
        vmin = np.nanpercentile(dt, 2)
        vmax = np.nanpercentile(dt, 98)
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        img = ax.imshow(dt, norm=norm, cmap="Greys")
        ax.set_aspect(1)
        ax.set_title("Set clip with mouse and press enter.")
        f.tight_layout()
        rect_selector = RectangleSelector(
            ax,
            onselect_function,
            useblit=True,
            button=[1, 3],
            interactive=True,
            minspanx=1,
            minspany=1,
            spancoords="pixels",
        )
        f.canvas.mpl_connect("key_press_event", on_press)
        plt.show()
        if not rect_selector.active:
            cmin, cmax, rmin, rmax = map(int, rect_selector.extents)
            sds = gdal.Translate(
                "/vsimem/in_memory_output.tif",
                gdal.Open(self.datasets["20m"]["xml"]),
                srcWin=[cmin, rmin, cmax - cmin, rmax - rmin],
            )
            ulx, xs, _, uly, _, ys = sds.GetGeoTransform()
            lrx = ulx + xs * sds.RasterXSize
            lry = uly + ys * sds.RasterYSize
            print(f"projWin=[{ulx}, {uly}, {lrx}, {lry}]")
            return self.clip((ulx, uly, lrx, lry), res=20, name=name, include8a=False)

    def warp_geojson(self, geojson, dstcrs=None, res=20, name="Clip", include8a=False, crop=False):
        """Reproject and clip scene by polygon stored in GeoJSON file

        Reproject all bands in scene to target CRS and clip to rectangle
        defined by bounds of polygonfeature  stored in GeoJSON file.
        If `crop` is True the bands are cropped to polygon outline. All bands
        in resulting collection have same resolution. For res=20, the '10m'
        dataset bands are downsampled, while for res=10, the bands of '20m'
        dataset are upsampled.

        Note:
            For GeoJSON without stored CRS information, the method assume that
            coordinates coincide with scene CRS

        Args:
            geojson (str): filename of GeoJSON with polygon feature

        Keyword Args:
            dstcrs(str or pyproj.CRS): coordinate system of clipped bands
            res (int): resolution of clipped bands. Default 20
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Dafault False
            crop (bool): whether to crop to polygon outline. Default False

        Returns:
            `s2lx.S2` collection

        """
        if dstcrs is not None:
            with open(geojson) as f:
                jsondata = json.load(f)
            if "crs" in jsondata:
                srcsrs = pyproj.CRS(jsondata["crs"]["properties"]["name"])
            else:
                srcsrs = self.crs
            dstcrs = pyproj.CRS(dstcrs)
            clip = geometry.shape(jsondata["features"][0]["geometry"]).buffer(0)
            if srcsrs != dstcrs:
                reproject = pyproj.Transformer.from_crs(srcsrs, dstcrs, always_xy=True).transform
                clip = ops.transform(reproject, clip)
            bnds = clip.bounds
            if crop:
                return self.warp(
                    bnds,
                    dstcrs=dstcrs,
                    name=name,
                    include8a=include8a,
                    cutlineLayer=geojson,
                )
            else:
                return self.warp(bnds, dstcrs=dstcrs, name=name, include8a=include8a)

    def warp(
        self,
        outputBounds,
        res=20,
        dstcrs=None,
        name="Clip",
        include8a=False,
        cutlineLayer=None,
    ):
        """Reproject and clip scene by by coordinates

        Reproject all bands in scene to target CRS and clip to rectangular
        window defined by coordinates. All bands in resulting collection have same
        resolution. For res=20, the '10m' dataset bands are downsampled,
        while for res=10, the bands of '20m' dataset are upsampled.

        Note:
            This method assume that bound coordinates are given in
            target CRS

        Args:
            outputBounds (tuple): output bounds as (minX, minY, maxX, maxY)
                in target CRS

        Keyword Args:
            res (int): resolution of clipped bands. Default 20
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Dafault False

        Returns:
            `s2lx.S2` collection

        """
        if outputBounds is not None and dstcrs is not None:
            wgs84 = pyproj.CRS("EPSG:4326")
            dstcrs = pyproj.CRS(dstcrs)
            reproject = pyproj.Transformer.from_crs(wgs84, dstcrs, always_xy=True).transform
            # safe footprint
            footprint = ops.transform(reproject, wkt.loads(self.meta["FOOTPRINT"])).buffer(-200)
            # go
            meta = self.meta.copy()
            rasters = []
            # 20m
            if cutlineLayer is not None:
                sds = gdal.Warp(
                    "/vsimem/in_memory_output.tif",
                    gdal.Open(self.datasets["20m"]["xml"]),
                    outputBounds=outputBounds,
                    dstSRS=dstcrs,
                    xRes=res,
                    yRes=res,
                    targetAlignedPixels=True,
                    cutlineLayer=cutlineLayer,
                    cropToCutline=True,
                    resampleAlg=gdal.GRA_Average,
                )
            else:
                sds = gdal.Warp(
                    "/vsimem/in_memory_output.tif",
                    gdal.Open(self.datasets["20m"]["xml"]),
                    outputBounds=outputBounds,
                    dstSRS=dstcrs,
                    xRes=res,
                    yRes=res,
                    targetAlignedPixels=True,
                    resampleAlg=gdal.GRA_Average,
                )
            # prepare mask
            transform = sds.GetGeoTransform()
            cg = np.arange(
                transform[0] + transform[1] // 2,
                transform[0] + sds.RasterXSize * transform[1],
                transform[1],
            )
            rg = np.arange(
                transform[3] + transform[5] // 2,
                transform[3] + sds.RasterYSize * transform[5],
                transform[5],
            )
            xx, yy = np.meshgrid(cg, rg)
            footpath = path.Path(np.array(footprint.exterior.xy).T)
            mask = np.invert(
                np.reshape(
                    footpath.contains_points(np.array([xx.ravel(), yy.ravel()]).T),
                    xx.shape,
                )
            )
            # cutline
            if cutlineLayer is not None:
                with open(cutlineLayer) as f:
                    jsondata = json.load(f)
                if "crs" in jsondata:
                    srcsrs = pyproj.CRS(jsondata["crs"]["properties"]["name"])
                else:
                    srcsrs = pyproj.CRS(self.datasets["20m"]["crs"])
                dstcrs = pyproj.CRS(dstcrs)
                clip = geometry.shape(jsondata["features"][0]["geometry"]).buffer(0)
                if srcsrs != dstcrs:
                    reproject = pyproj.Transformer.from_crs(srcsrs, dstcrs, always_xy=True).transform
                    clip = ops.transform(reproject, clip)
                clippath = path.Path(np.array(clip.exterior.xy).T)
                mask = np.logical_or(
                    mask,
                    np.invert(
                        np.reshape(
                            clippath.contains_points(np.array([xx.ravel(), yy.ravel()]).T),
                            xx.shape,
                        )
                    ),
                )
            # go
            for band in ["B5", "B6", "B7", "B8A", "B11", "B12"]:
                bmeta = self.datasets["20m"]["bands"][band].copy()
                rb = sds.GetRasterBand(bmeta.pop("n"))
                vals = rb.ReadAsArray().astype(np.int16)
                if band == "B8A":
                    if include8a:
                        rasters.append(
                            Band(
                                "a8",
                                np.ma.masked_array(vals, mask),
                                sds.GetGeoTransform(),
                                sds.GetProjection(),
                            )
                        )
                        meta["a8"] = bmeta
                else:
                    rasters.append(
                        Band(
                            band.lower(),
                            np.ma.masked_array(vals, mask),
                            sds.GetGeoTransform(),
                            sds.GetProjection(),
                        )
                    )
                    meta[band.lower()] = bmeta
            # 10 m
            if cutlineLayer is not None:
                sds = gdal.Warp(
                    "/vsimem/in_memory_output.tif",
                    gdal.Open(self.datasets["10m"]["xml"]),
                    outputBounds=outputBounds,
                    dstSRS=dstcrs,
                    xRes=res,
                    yRes=res,
                    targetAlignedPixels=True,
                    cutlineLayer=cutlineLayer,
                    cropToCutline=True,
                    resampleAlg=gdal.GRA_Average,
                )
            else:
                sds = gdal.Warp(
                    "/vsimem/in_memory_output.tif",
                    gdal.Open(self.datasets["10m"]["xml"]),
                    outputBounds=outputBounds,
                    dstSRS=dstcrs,
                    xRes=res,
                    yRes=res,
                    targetAlignedPixels=True,
                    resampleAlg=gdal.GRA_Average,
                )
            for band in ["B4", "B3", "B2", "B8"]:
                bmeta = self.datasets["10m"]["bands"][band].copy()
                rb = sds.GetRasterBand(bmeta.pop("n"))
                vals = rb.ReadAsArray().astype(np.int16)
                rasters.append(
                    Band(
                        band.lower(),
                        np.ma.masked_array(vals, mask),
                        sds.GetGeoTransform(),
                        sds.GetProjection(),
                    )
                )
                meta[band.lower()] = bmeta
            return S2(*rasters, meta=meta, name=name)


class S2:
    """Class to store homogeneous collection of bands

    All bands in collection share geographic reference, size and resolution.
    Individual bands in collection could be accessed by dot notation.

    Args:
        *args: any number of `Band` instances

    Keyword Args:
        name (str): mame of collection. Default is 'S2'
        meta (dict): Dictionary with metadata. Default is {}

    Examples:
        To acceess band, just use band name

        >>> d.bands
        ['b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']
        >>> d.b12 / d.b8
        Band b12/b8 (904, 1041) float64
        Min:0.500583 Max:1.86126 Vmin:0.962256 Vmax:1.40753

    """

    def __init__(self, *rasters, **kwargs):
        self._bands = rasters
        self.meta = kwargs.get("meta", {})
        self.name = kwargs.get("name", "S2")
        for b in rasters:
            self.__dict__[b.name] = b

    def __repr__(self):
        return f"{self.name} Bands: {self.bands}"

    @property
    def bands(self):
        """list: Sorted list of band names"""
        return sorted([k.name for k in self._bands])

    def restored_pca(self, **kwargs):
        """PCA based filtering"""
        remove = kwargs.get("remove", ())
        if isinstance(remove, int):
            remove = (remove,)
        X = np.array([b.values for b in self._bands]).T
        pca = decomposition.PCA(n_components=len(self._bands))
        y_pred = pca.fit_transform(X)
        for r in remove:
            y_pred[:, r] = 0
        restored = pca.inverse_transform(y_pred)
        rasters = []
        for ix, b in enumerate(self._bands):
            vals = b.array.copy()
            vals[np.invert(b.array.mask)] = restored[:, ix]
            rasters.append(b.copy(array=vals))
        meta = dict(
            components=pca.components_,
            ev=pca.explained_variance_,
            evr=pca.explained_variance_ratio_,
        )
        return S2(*rasters, meta=meta, name=self.name + " PCA restored")

    def pca(self, **kwargs):
        """PCA analysis"""
        centered = kwargs.get("centered", True)
        n = kwargs.get("n", len(self._bands))
        X = np.array([b.values for b in self._bands]).T
        if centered:
            X -= X.mean(axis=0)
        pca = decomposition.PCA(n_components=n)
        y_pred = pca.fit_transform(X)
        rasters = []
        for ix in range(n):
            vals = self._bands[0].array.copy()
            vals[np.invert(self._bands[0].array.mask)] = y_pred[:, ix]
            rasters.append(self._bands[0].copy(array=vals, name=f"pc{ix}"))
        meta = dict(
            components=pca.components_,
            ev=pca.explained_variance_,
            evr=pca.explained_variance_ratio_,
        )
        return S2(*rasters, meta=meta, name=self.name + " PCA")


class Band:
    """Class to store band data

    Band is internally stores as masked array tu support ROI operations

    Args:
        name (str): name of the band. Name must start with a letter or the
            underscore character (not a number) and can only contain
            alpha-numeric characters and underscores.
        array (numpy.ma.array): band data as 2d numpy masked array
        transform (tuple): 6 coefficients geotransform. Rotation not supported
        projection (str): CRS in WKT

    Attributes:
        shape (tuple): shape of raster array
        vmin (float): minimum value used for colorscale. Default 2% percentile
        vmax (float): maximum value used for colorscale. Default 98% percentile

    """

    def __init__(self, name, array, transform, projection):
        self.array = array
        self.name = name
        self.transform = transform
        self.projection = projection
        self.shape = array.shape
        self.vmin = np.nanpercentile(self.values, 2)
        self.vmax = np.nanpercentile(self.values, 98)

    def __repr__(self):
        txt = f"Band {self.name} {self.shape} {self.dtype}\n"
        txt += f"Min:{self.min:g} Max:{self.max:g} Vmin:{self.vmin:g} Vmax:{self.vmax:g}"
        return txt

    def copy(self, **kwargs):
        name = kwargs.get("name", self.name)
        array = kwargs.get("array", self.array).copy()
        return Band(name, array, self.transform, self.projection)

    @property
    def values(self):
        return self.array[np.invert(self.array.mask)].data

    def __array__(self, dtype=None):
        if dtype is None:
            return self.array
        else:
            return self.array.astype(dtype)

    def __add__(self, other):
        if isinstance(other, Band):
            return self.copy(array=self.array + other.array, name=f"{self.name}+{other.name}")
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array + other, name=f"{self.name}+{other}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __radd__(self, other):
        if isinstance(other, Band):
            return self.copy(array=self.array + other.array, name=f"{other.name}+{self.name}")
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array + other, name=f"{other}+{self.name}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __sub__(self, other):
        if isinstance(other, Band):
            return self.copy(array=self.array - other.array, name=f"{self.name}-{other.name}")
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array - other, name=f"{self.name}-{other}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __rsub__(self, other):
        if isinstance(other, Band):
            return self.copy(array=other.array - self.array, name=f"{other.name}-{self.name}")
        elif isinstance(other, numbers.Number):
            return self.copy(array=other - self.array, name=f"{other}-{self.name}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __mul__(self, other):
        if "+" in self.name or "-" in self.name:
            sname = f"({self.name})"
        else:
            sname = self.name
        if isinstance(other, Band):
            if "+" in other.name or "-" in other.name:
                oname = f"({other.name})"
            else:
                oname = other.name
            return self.copy(array=self.array * other.array, name=f"{sname}*{oname}")
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array * other, name=f"{sname}*{other}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __rmul__(self, other):
        if "+" in self.name or "-" in self.name:
            sname = f"({self.name})"
        else:
            sname = self.name
        if isinstance(other, Band):
            if "+" in other.name or "-" in other.name:
                oname = f"({other.name})"
            else:
                oname = other.name
            return self.copy(array=self.array * other.array, name=f"{oname}*{sname}")
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array * other, name=f"{other}*{sname}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __truediv__(self, other):
        if "+" in self.name or "-" in self.name:
            sname = f"({self.name})"
        else:
            sname = self.name
        if isinstance(other, Band):
            if "+" in other.name or "-" in other.name:
                oname = f"({other.name})"
            else:
                oname = other.name
            return self.copy(array=self.array / other.array, name=f"{sname}/{oname}")
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array / other, name=f"{sname}/{other}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __rtruediv__(self, other):
        if "+" in self.name or "-" in self.name:
            sname = f"({self.name})"
        else:
            sname = self.name
        if isinstance(other, Band):
            if "+" in other.name or "-" in other.name:
                oname = f"({other.name})"
            else:
                oname = other.name
            return self.copy(array=self.array / other.array, name=f"{oname}/{sname}")
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array / other, name=f"{other}/{sname}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __abs__(self):
        return Band(f"|{self.name}|", abs(self.array))

    @property
    def dtype(self):
        return self.array.dtype

    def astype(self, dtype):
        return self.copy(array=self.array.astype(dtype))

    @property
    def min(self):
        return self.array.min()

    @property
    def max(self):
        return self.array.max()

    @property
    def norm(self):
        return colors.Normalize(vmin=self.vmin, vmax=self.vmax, clip=True)

    @property
    def normalized(self):
        return self.norm(self.array)

    def apply(self, fun):
        sname = f"{fun.__name__}" + f"({self.name})"
        vals = self.array.copy()
        vals[np.invert(self.array.mask)] = fun(self.values)
        return self.copy(array=vals, name=sname)

    def patch(self, other, fit=False):
        assert self.transform == other.transform, "transform must be identical"
        assert self.projection == other.projection, "projection must be identical"
        vals = np.array(self.array)
        mask = np.array(self.array.mask)
        ovals = np.array(other.array)[mask]
        if fit:
            o1, o2 = self.intersect(other)
            pp = np.polyfit(o2.values, o1.values, 1)
            ovals = ovals * pp[0] + pp[1]
        vals[mask] = ovals
        mask[mask] = np.array(other.array.mask)[mask]
        return self.copy(array=np.ma.masked_array(vals, mask))

    def intersect(self, other):
        assert self.transform == other.transform, "transform must be identical"
        assert self.projection == other.projection, "projection must be identical"
        mask1 = np.array(self.array.mask)
        mask2 = np.array(other.array.mask)
        mask = np.logical_and(np.invert(mask1), np.invert(mask2))
        b1 = self.copy()
        b1.array.mask = np.invert(mask)
        b2 = other.copy()
        b2.array.mask = np.invert(mask)
        return b1, b2

    def show(self, **kwargs):
        # parse kwargs
        figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        cmap = plt.get_cmap(kwargs.get("cmap", "cividis"))
        cmap.set_under(kwargs.get("under", cmap(0.0)))
        cmap.set_over(kwargs.get("over", cmap(1.0)))
        cmap.set_bad(kwargs.get("masked", "white"))
        # plot
        f, ax = plt.subplots(figsize=figsize)
        img = ax.imshow(self.array, cmap=cmap, norm=self.norm)
        ax.set_title(f"{self.name}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        f.colorbar(img, cax=cax, extend="both")
        ax.set_aspect(1)
        f.tight_layout()
        plt.show()

    def save(self, filename, **kwargs):
        # save
        if filename.lower().endswith(".tif"):
            driver_gtiff = gdal.GetDriverByName("GTiff")
            ds_create = driver_gtiff.Create(
                filename,
                xsize=self.shape[1],
                ysize=self.shape[0],
                bands=1,
                eType=gdal_array.NumericTypeCodeToGDALTypeCode(self.dtype),
            )
            ds_create.SetProjection(self.projection)
            ds_create.SetGeoTransform(self.transform)
            dt = np.array(self.array.data)
            if np.any(self.array.mask):
                dt[self.array.mask] = 0
                ds_create.GetRasterBand(1).SetNoDataValue(0)
            ds_create.GetRasterBand(1).WriteArray(dt)
            ds_create = None
        else:
            print("filename must have .tif extension")


class Composite:
    """Class to store RGB composite

    Args:
        r (Band): band used for red channel
        g (Band): band used for green channel
        b (Band): band used for blue channel

    Keyword Args:
        name (str): name of the RGB composite

    Attributes:
        shape (tuple): shape of raster array
        transform (tuple): 6 coefficients geotransform. Rotation not supported
        projection (str): CRS in WKT

    """

    def __init__(self, r, g, b, name="RGB composite"):
        assert r.shape == g.shape == b.shape, "Shapes of bands must be same"
        self.name = name
        self.r = r
        self.g = g
        self.b = b
        self.transform = r.transform
        self.projection = r.projection
        self.shape = r.shape

    def __repr__(self):
        return f"Composite {self.name} {self.shape} [{self.r.name} {self.g.name} {self.b.name}]"

    def __array__(self, dtype=None):
        if dtype is None:
            return np.dstack((self.r, self.g, self.b))
        else:
            return np.dstack((self.r, self.g, self.b)).astype(dtype)

    def show(self, **kwargs):
        # parse kwargs
        figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        rgb = np.dstack((self.r.normalized, self.g.normalized, self.b.normalized))
        # plot
        f, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.set_title(f"{self.name} [{self.r.name} {self.g.name} {self.b.name}]")
        ax.set_aspect(1)
        f.tight_layout()
        plt.show()

    def save(self, filename, **kwargs):
        # save
        if filename.lower().endswith(".tif"):
            driver_gtiff = gdal.GetDriverByName("GTiff")
            options = ["PHOTOMETRIC=RGB", "PROFILE=GeoTIFF"]
            ds_create = driver_gtiff.Create(
                filename,
                xsize=self.shape[1],
                ysize=self.shape[0],
                bands=4,
                eType=gdal.GDT_Float64,
                options=options,
            )
            ds_create.SetProjection(self.projection)
            ds_create.SetGeoTransform(self.transform)
            # red
            ds_create.GetRasterBand(1).WriteArray(np.array(self.r.normalized))
            ds_create.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
            # green
            ds_create.GetRasterBand(2).WriteArray(np.array(self.g.normalized))
            ds_create.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
            # blue
            ds_create.GetRasterBand(3).WriteArray(np.array(self.b.normalized))
            ds_create.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
            # aplha
            ds_create.GetRasterBand(4).WriteArray(255 * np.invert(self.r.array.mask))
            ds_create.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
            ds_create = None
        else:
            print("filename must have .tif extension")
