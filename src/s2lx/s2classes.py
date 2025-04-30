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
from osgeo import gdal, gdal_array, ogr
from shapely import geometry, ops, wkt
from sklearn import decomposition
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

gdal.UseExceptions()
ogr.UseExceptions()


class SAFEStore:
    """Class to manipulate Sentinel-2 product store

    The SAFEStore is a folder where all SAFE products ares stored.

    Attributes:
        path (Path): Path to directory
        SAFES (list): List of all SAFE directories in store
        tiles (list): List of all tile numbers in store
    """

    def __init__(self, path):
        self.path = Path(path)
        assert len(self.SAFES) > 0, "No SAFE found in directory."

    @property
    def SAFES(self):
        return list(self.path.glob("*.SAFE"))

    @property
    def tiles(self):
        return [s.stem.split("_")[5] for s in self.SAFES]

    def getsafe(self, tile):
        candidates = list(self.path.glob(f"*_{tile}_*"))
        if len(candidates) > 0:
            return SAFE(list(candidates[0].glob("MTD_*"))[0])

    def searchsafe(self, filename, driverName="GeoJSON"):
        for safe in self.path.iterdir():
            s = SAFE(list(safe.glob("MTD_*"))[0])
            r = s.coverage(filename, driverName=driverName)
            if r > 0:
                cl = float(s.meta["CLOUD_COVERAGE_ASSESSMENT"]) / 100
                print(
                    safe.stem.split("_")[5],
                    f"{r:>6.2%}",
                    s.meta["PRODUCT_START_TIME"],
                    f"{cl:>7.3%}",
                    s.meta["PROCESSING_LEVEL"],
                    s.crs.to_string(),
                )

    def warp_patch(
        self,
        filename,
        driverName="GeoJSON",
        res=20,
        name="Clip",
        include8a=False,
        crop=True,
        tiles=None,
        fit=False,
    ):
        if tiles is None:
            rat = []
            safes = []
            for safe in self.path.iterdir():
                s = SAFE(list(safe.glob("MTD_*"))[0])
                r = s.coverage(filename, driverName=driverName)
                if r > 0:
                    rat.append(r)
                    safes.append(s)
            ixs = np.argsort(rat)
        else:
            safes = [self.getsafe(t) for t in tiles]
            ixs = list(range(len(safes)))
            rat = [s.coverage(filename, driverName=driverName) for s in safes]
        # do
        print(f"Reading {safes[ixs[0]].tilenumber} [{rat[ixs[0]]:.2%}]...")
        d = safes[ixs[0]].warp_features(
            filename,
            driverName=driverName,
            res=res,
            name=name,
            include8a=include8a,
            crop=crop,
        )
        n = 1
        for ix in ixs[1:]:
            print(
                f"{n}/{len(ixs) - 1} Patching {safes[ix].tilenumber} [{rat[ix]:.2%}]..."
            )
            other = safes[ix].warp_features(
                filename,
                driverName=driverName,
                res=res,
                name=name,
                include8a=include8a,
                crop=crop,
            )
            d = d.patch(other, fit=fit)
            n += 1
        return d


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
                dsname, crs = xmlfile.split(".xml:")[1].split(":")
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
        txt += f'{self.meta["PROCESSING_LEVEL"]} '
        txt += f"{self.tilenumber}\n"
        txt += f'Sensing date {self.meta["DATATAKE_1_DATATAKE_SENSING_START"]}\n'
        for d in self.datasets:
            txt += (
                f'Dataset {d} bands: {sorted(list(self.datasets[d]["bands"].keys()))}\n'
            )
        return txt

    @property
    def tilenumber(self):
        return self.meta["PRODUCT_URI"].split("_")[5]

    def preview(self, **kwargs):
        """Show the scene TCI image

        Keyword Args:
            figsize (tuple): matplotlib Figure size
            filename(str): if not None, the plot is save to file.
                Default None
            dpi (int): DPI for save image. Default 300

        """
        sds = gdal.Open(self.datasets["TCI"]["xml"])
        r = sds.GetRasterBand(self.datasets["TCI"]["bands"]["B4"]["n"]).ReadAsArray()
        g = sds.GetRasterBand(self.datasets["TCI"]["bands"]["B3"]["n"]).ReadAsArray()
        b = sds.GetRasterBand(self.datasets["TCI"]["bands"]["B2"]["n"]).ReadAsArray()
        tci = np.dstack((r, g, b))
        transform = sds.GetGeoTransform()
        projection = sds.GetProjection()
        figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        # plot
        f, ax = plt.subplots(figsize=figsize)
        extent = (
            transform[0],
            transform[0] + r.shape[1] * transform[1],
            transform[3] + r.shape[0] * transform[5],
            transform[3],
        )
        ax.imshow(tci, extent=extent)
        ax.set_title(
            f"{self.meta['PRODUCT_URI'].split('_')[5]} {pyproj.CRS(projection).name}"
        )
        ax.set_aspect(1)
        filename = kwargs.get("filename", None)
        f.tight_layout()
        if filename is not None:
            f.savefig(filename, dpi=kwargs.get("dpi", 300))
        plt.show()

    @property
    def crs(self):
        """pyproj.CRS: coordinate reference system of scene"""
        return pyproj.CRS(self.datasets["20m"]["crs"])

    def footprint(self, dstcrs=None):
        """Get scene footprint

        Keyword Args:
            dstcrs(str or pyproj.CRS): coordinate system of footprint.
                Default is scene CRS.

        Returns:
            Shapely polygon feature

        """
        wgs84 = pyproj.CRS("EPSG:4326")
        if dstcrs is None:
            dstcrs = self.crs
        else:
            dstcrs = pyproj.CRS(dstcrs)
        fpoly = wkt.loads(self.meta["FOOTPRINT"])
        if dstcrs != wgs84:
            reproject = pyproj.Transformer.from_crs(
                wgs84, dstcrs, always_xy=True
            ).transform
            # safe footprint
            fpoly = ops.transform(reproject, fpoly).buffer(0)
        return fpoly

    def overlap(self, other):
        """Check if scene footprints overlap

        Args:
            other: `s2lx.SAFE` scene

        Returns:
            True is footprints overlap, otherwise False

        """
        wgs84 = pyproj.CRS("EPSG:4326")
        return self.footprint(dstcrs=wgs84).intersects(other.footprint(dstcrs=wgs84))

    def intersection(self, other, dstcrs=None):
        """Return intersections of scene footprints as shapely polygon

        Args:
            other: `s2lx.SAFE` scene

        Keyword Args:
            dstcrs(str or pyproj.CRS): coordinate system for intersection.
                Default is scene CRS.

        Returns:
            shapely polygon

        """
        wgs84 = pyproj.CRS("EPSG:4326")
        if dstcrs is None:
            dstcrs = self.crs
        else:
            dstcrs = pyproj.CRS(dstcrs)
        inter = self.footprint(dstcrs=wgs84).intersection(other.footprint(dstcrs=wgs84))
        if wgs84 != dstcrs:
            reproject = pyproj.Transformer.from_crs(
                wgs84, dstcrs, always_xy=True
            ).transform
            inter = ops.transform(reproject, inter)
        return inter

    def coverage(self, filename, driverName="GeoJSON"):
        """Return area fraction of features in filename covered by scene

        Args:
            filename (str): filename of vector file

        Keyword Args:
            driverName (str): format of vector file. Default is 'GeoJSON'
                For available options see `ogr2ogr --formats`

        Returns:
            float value of coverage (between 0 and 1)

        """
        driver = ogr.GetDriverByName(driverName)
        ds = driver.Open(filename, 0)
        layer = ds.GetLayer()
        clip = ops.unary_union(
            [wkt.loads(f.GetGeometryRef().ExportToWkt()).buffer(0) for f in layer]
        )
        try:
            crs = pyproj.CRS.from_wkt(layer.GetSpatialRef().ExportToWkt())
        except pyproj.exceptions.CRSError:
            crs = pyproj.CRS("EPSG:4326")
        fp = self.footprint(dstcrs=crs)
        if clip.intersects(fp):
            rat = clip.intersection(fp).area / clip.area
        else:
            rat = 0
        return rat

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

    def clip_features(
        self, filename, driverName="GeoJSON", res=20, name="Clip", include8a=False
    ):
        """Clip scene to features extent in vector file

        Clip all bands in scene by rectangular extent of features
        stored in vector file. All bands in resulting collection have same
        resolution. For res=20, the '10m' dataset bands are downsampled,
        while for res=10, the bands of '20m' dataset are upsampled.

        Note:
            This method assume that CRS of vector file and scene is identical

        Args:
            filename (str): filename of vector file

        Keyword Args:
            driverName (str): format of vector file. Default is 'GeoJSON'
                For available options see `ogr2ogr --formats`
            res (int): resolution of clipped bands. Default 20
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Dafault False

        Returns:
            `s2lx.S2` collection

        """
        driver = ogr.GetDriverByName(driverName)
        ds = driver.Open(filename, 0)
        layer = ds.GetLayer()
        #
        try:
            srccrs = pyproj.CRS.from_wkt(layer.GetSpatialRef().ExportToWkt())
        except pyproj.exceptions.CRSError:
            srccrs = self.crs
        dstcrs = self.crs
        # to do
        clip = ops.unary_union(
            [wkt.loads(f.GetGeometryRef().ExportToWkt()).buffer(0) for f in layer]
        )
        if srccrs != dstcrs:
            reproject = pyproj.Transformer.from_crs(
                srccrs, dstcrs, always_xy=True
            ).transform
            clip = ops.transform(reproject, clip)
        #
        # extent = layer.GetExtent()
        # bounds = (extent[0], extent[2], extent[1], extent[3])
        bounds = clip.bounds
        return self.clip(
            bounds,
            res=res,
            name=name,
            include8a=include8a,
        )

    def clip_shape(self, shape, srccrs=None, res=20, name="Clip", include8a=False):
        """Clip scene to features extent in vector file

        Clip all bands in scene by bounding box of shapely polygon. All bands
        in resulting collection have same resolution. For res=20, the '10m'
        dataset bands are downsampled, while for res=10, the bands of '20m'
        dataset are upsampled.

        Args:
            shape: shapely polygon

        Keyword Args:
            srccrs(str or pyproj.CRS): shape coordinate system. Default
                is CRS of scene
            res (int): resolution of clipped bands. Default 20
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Dafault False

        Returns:
            `s2lx.S2` collection

        """
        if srccrs is None:
            srccrs = self.crs
        else:
            srccrs = pyproj.CRS(srccrs)
        if srccrs != self.crs:
            reproject = pyproj.Transformer.from_crs(
                srccrs, self.crs, always_xy=True
            ).transform
            shape = ops.transform(reproject, shape)
        return self.clip(
            shape.bounds,
            res=res,
            name=name,
            include8a=include8a,
        )

    def clip(self, bounds, res=20, name="Clip", include8a=False):
        """Clip scene by extent coordinates

        Clip all bands in scene by rectangular bound (minX, minY, maxX, maxY).
        All bands in resulting collection have same resolution. For res=20,
        the '10m' dataset bands are downsampled, while for res=10, the bands
        of '20m' dataset are upsampled.

        Note:
            Coordinates of the extent of the output are automatically aligned
            to multiples of resolution. This method also assume that coordinates
            of subwindow are given in CRS of scene

        Args:
            bounds (tuple): output bounds as (minX, minY, maxX, maxY)
                in target CRS

        Keyword Args:
            res (int): resolution of clipped bands. Default 20
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Dafault False

        Returns:
            `s2lx.S2` collection

        """
        footprint = self.footprint().buffer(-200)
        # aligned pixels
        bounds = [res * np.round(v / res) for v in bounds]
        # prepare mask
        cg = np.arange(bounds[0] + res // 2, bounds[2], res)
        rg = np.arange(bounds[3] - res // 2, bounds[1], -res)
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
            projWin=(bounds[0], bounds[3], bounds[2], bounds[1]),
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
            projWin=(bounds[0], bounds[3], bounds[2], bounds[1]),
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
        ax.imshow(dt, norm=norm, cmap="Greys")
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
            minX, xs, _, maxY, _, ys = sds.GetGeoTransform()
            maxX = minX + xs * sds.RasterXSize
            minY = maxY + ys * sds.RasterYSize
            print(f"bounds=({minX}, {minY}, {maxX}, {maxY})")
            return self.clip(
                (minX, minY, maxX, maxY), res=20, name=name, include8a=False
            )

    def warp_features(
        self,
        filename,
        dstcrs=None,
        driverName="GeoJSON",
        res=20,
        name="Clip",
        include8a=False,
        crop=True,
    ):
        """Reproject and clip scene to extent of features in vector file

        Reproject all bands in scene to target CRS and clip to rectangular region
        defined by extent of features in vector file. If `crop` is True, the bands
        are cropped to outline of features. All bands in resulting collection have
        same resolution. For res=20, the '10m' dataset bands are downsampled, while
        for res=10, the bands of '20m' dataset are upsampled.

        Note:
            For vector formats without stored CRS information, the method assume that
            coordinates coincide with scene CRS

        Args:
            filename (str): filename of vector file

        Keyword Args:
            driverName (str): format of vector file. Default is 'GeoJSON'
                For available options see `ogr2ogr --formats`
            dstcrs(str or pyproj.CRS): target coordinate system. Default
                is CRS of vector file
            res (int): resolution of clipped bands. Default 20
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Default False
            crop (bool): whether to crop to polygon outline. Default False

        Returns:
            `s2lx.S2` collection

        """
        driver = ogr.GetDriverByName(driverName)
        ds = driver.Open(filename, 0)
        layer = ds.GetLayer()
        try:
            srccrs = pyproj.CRS.from_wkt(layer.GetSpatialRef().ExportToWkt())
        except pyproj.exceptions.CRSError:
            srccrs = self.crs
        if dstcrs is None:
            dstcrs = srccrs
        else:
            dstcrs = pyproj.CRS(dstcrs)
        # to do
        clip = ops.unary_union(
            [wkt.loads(f.GetGeometryRef().ExportToWkt()).buffer(0) for f in layer]
        )
        if srccrs != dstcrs:
            reproject = pyproj.Transformer.from_crs(
                srccrs, dstcrs, always_xy=True
            ).transform
            clip = ops.transform(reproject, clip)
        if crop:
            return self.warp(
                clip.bounds,
                dstcrs,
                name=name,
                include8a=include8a,
                cutlineLayer=filename,
                driverName=driverName,
            )
        else:
            return self.warp(clip.bounds, dstcrs, name=name, include8a=include8a)

    def warp_shape(
        self, shape, srccrs=None, dstcrs=None, res=20, name="Clip", include8a=False
    ):
        """Reproject and clip scene to extent of shapely polygon

        Reproject all bands in scene to target CRS and clip to bounding box of
        shapely polygon. All bands in resulting collection have same resolution.
        For res=20, the '10m' dataset bands are downsampled, while for res=10,
        the bands of '20m' dataset are upsampled.

        Args:
            shape (str): shapely polygon

        Keyword Args:
            srccrs(str or pyproj.CRS): shape coordinate system. Default
                is CRS of scene
            dstcrs(str or pyproj.CRS): target coordinate system. Default
                is srccrs
            res (int): resolution of clipped bands. Default 20
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Default False

        Returns:
            `s2lx.S2` collection

        """
        if srccrs is None:
            srccrs = self.crs
        else:
            srccrs = pyproj.CRS(srccrs)
        if dstcrs is None:
            dstcrs = srccrs
        else:
            dstcrs = pyproj.CRS(dstcrs)
        # to do
        if srccrs != dstcrs:
            reproject = pyproj.Transformer.from_crs(
                srccrs, dstcrs, always_xy=True
            ).transform
            shape = ops.transform(reproject, shape)
        return self.warp(shape.bounds, dstcrs, name=name, include8a=include8a)

    def warp(
        self,
        bounds,
        dstcrs,
        res=20,
        name="Clip",
        include8a=False,
        cutlineLayer=None,
        driverName="GeoJSON",
    ):
        """Reproject and clip scene by by coordinates

        Reproject all bands in scene to target CRS and clip to rectangular
        window defined by coordinates. All bands in resulting collection have same
        resolution. For res=20, the '10m' dataset bands are downsampled,
        while for res=10, the bands of '20m' dataset are upsampled.

        Note:
            Coordinates of the extent of the output are automatically aligned
            to multiples of resolution. This method also assume that bound
            coordinates are given in target CRS.

        Args:
            bounds (tuple): output bounds as (minX, minY, maxX, maxY)
                in target CRS
            dstcrs(str or pyproj.CRS): target coordinate system

        Keyword Args:
            res (int): resolution of clipped bands. Default 20
            name (str): name of collection. Default is 'Clip'
            include8a (bool): whether to include B8A band. Dafault False
            cutlineLayer (str): filename of vector file. Default None
                All pixels out of cutlineLayer are masked
            driverName (str): format of vector file. Default is 'GeoJSON'
                For available options see `ogr2ogr --formats`

        Returns:
            `s2lx.S2` collection

        """
        dstcrs = pyproj.CRS(dstcrs)
        footprint = self.footprint(dstcrs=dstcrs).buffer(-200)
        # aligned pixels
        bounds = [res * np.round(v / res) for v in bounds]
        # go
        meta = self.meta.copy()
        rasters = []
        # 20m
        if cutlineLayer is not None:
            sds = gdal.Warp(
                "/vsimem/in_memory_output.tif",
                gdal.Open(self.datasets["20m"]["xml"]),
                outputBounds=bounds,
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
                outputBounds=bounds,
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
            driver = ogr.GetDriverByName(driverName)
            ds = driver.Open(cutlineLayer, 0)
            layer = ds.GetLayer()
            try:
                srccrs = pyproj.CRS.from_wkt(layer.GetSpatialRef().ExportToWkt())
            except pyproj.exceptions.CRSError:
                srccrs = self.crs
            clip = ops.unary_union(
                [wkt.loads(f.GetGeometryRef().ExportToWkt()).buffer(0) for f in layer]
            )
            if srccrs != dstcrs:
                reproject = pyproj.Transformer.from_crs(
                    srccrs, dstcrs, always_xy=True
                ).transform
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
                outputBounds=bounds,
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
                outputBounds=bounds,
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

    Attributes:
        transform (tuple): 6 coefficients geotransform. Rotation not supported
        projection (str): CRS in WKT

    Examples:
        To acceess band, just use band name

        >>> d.bands
        ['b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']
        >>> d.b12 / d.b8
        Band b12/b8 (904, 1041) float64
        Min:0.500583 Max:1.86126 Vmin:0.962256 Vmax:1.40753

    """

    def __init__(self, *rasters, **kwargs):
        assert (
            len(set([b.transform for b in rasters])) == 1
        ), "All bands must have same transform"
        assert (
            len(set([b.projection for b in rasters])) == 1
        ), "All bands must have same projection"
        self._bands = rasters
        self.meta = kwargs.get("meta", {})
        self.name = kwargs.get("name", "S2")
        self.transform = rasters[0].transform
        self.projection = rasters[0].projection
        for b in rasters:
            self.__dict__[b.name] = b

    def __repr__(self):
        return f"{self.name} Bands: {self.bands}"

    @property
    def bands(self):
        """list: Sorted list of band names"""
        return sorted([k.name for k in self._bands])

    def patch(self, other, fit=False):
        """Patch collection

        All masked regions are patched from other

        Args:
            other: `s2lx.S2` collection

        Keyword Args:
            fit (bool): If True, the patching dataset bands are linearly
                scaled to fit in overlapping region. Default False

        Returns:
            `s2lx.S2` collection

        """
        assert self.transform == other.transform, "transform must be identical"
        assert self.projection == other.projection, "projection must be identical"

        return S2(
            *(a.patch(b, fit=fit) for (a, b) in zip(self._bands, other._bands)),
            meta=self.meta,
            name=f"{self.name}+{other.name}",
        )

    def restored_pca(self, **kwargs):
        """PCA based filtering

        Use PCA components with cumulative explained variance given
        by treshold. Aternatively, PCA components to be excluded from
        reconstruction could be defined.

        Keyword Args:
            remove (int or list): PCA components to be removed
            threshold (float): Threshold of explained variance
                in percents to be reconstructed. Default 98.

        Returns:
            `s2lx.S2` collection. Components, explained variance and
                explained variance ratio are available in metadata

        """
        X = np.array([b.values for b in self._bands]).T
        pca = decomposition.PCA(n_components=len(self._bands))
        y_pred = pca.fit_transform(X)
        # manage kwargs
        remove = kwargs.get("remove", None)
        if remove is None:
            threshold = kwargs.get("threshold", 98)
            last_c = np.where(
                100 * np.cumsum(pca.explained_variance_ratio_) > threshold
            )[0][0]
            print(f"Using {last_c + 1} components")
            remove = np.arange(0, last_c + 1)
        if isinstance(remove, int):
            remove = (remove,)
        # go
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
        """PCA analysis

        Calculate principal components from all bands in collection

        Keyword Args:
            centered (bool): If True values are centered
            n (int): Number of pricipal components calculated.
                Default is number of bands.

        Returns:
            `s2lx.S2` collection. Components, explained variance and
                explained variance ratio are available in metadata

        """
        centered = kwargs.get("centered", True)
        n = kwargs.get("n", len(self._bands))
        X = np.array([b.values for b in self._bands]).T
        if centered:
            X = X - X.mean(axis=0)
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

    def truecolor(self, name="True Color"):
        assert "b4" in self.__dict__, "Band 4 not present"
        assert "b3" in self.__dict__, "Band 3 not present"
        assert "b2" in self.__dict__, "Band 2 not present"
        return Composite(self.b4, self.b3, self.b2, name=name)

    def falsecolor(self, name="False Color"):
        assert "b12" in self.__dict__, "Band 12 not present"
        assert "b11" in self.__dict__, "Band 11 not present"
        assert "b8" in self.__dict__, "Band 8 not present"
        return Composite(self.b12, self.b11, self.b8, name=name)

    def georatio(self, name="GeoRatio"):
        assert "b12" in self.__dict__, "Band 12 not present"
        assert "b11" in self.__dict__, "Band 11 not present"
        assert "b8" in self.__dict__, "Band 8 not present"
        assert "b4" in self.__dict__, "Band 4 not present"
        assert "b3" in self.__dict__, "Band 3 not present"
        assert "b2" in self.__dict__, "Band 2 not present"
        return Composite(
            self.b12 / self.b4, self.b11 / self.b3, self.b8 / self.b2, name=name
        )

    def alteration(self, name="Alterations"):
        assert "b12" in self.__dict__, "Band 12 not present"
        assert "b11" in self.__dict__, "Band 11 not present"
        assert "b4" in self.__dict__, "Band 4 not present"
        assert "b2" in self.__dict__, "Band 2 not present"
        return Composite(
            self.b11 / self.b12, self.b4 / self.b2, self.b4 / self.b11, name=name
        )

    def showpca(self):
        assert "evr" in self.meta, "This is no PCA S2 dataset"
        f, axs = plt.subplots(3, 3, figsize=(14, 9))
        for b, ax, r in zip(self._bands, axs.flatten(), self.meta["evr"]):
            b.show(ax=ax, showproj=False)
            ax.set_title(f"{b.name}:{100 * r:g}")
        f.tight_layout()
        plt.show()


class Band:
    """Class to store band data

    Raster band, internally stored as masked array, to support ROI operations.
    Bands support basic mathematical operations.

    Args:
        name (str): name of the band. Name must start with a letter or the
            underscore character (not a number) and can only contain
            alpha-numeric characters and underscores.
        array (numpy.ma.array): band data as 2d numpy masked array
        transform (tuple): 6 coefficients geotransform. Rotation not supported
        projection (str): CRS in WKT

    Keyword Args:
        vmin (float): minimum value for color normalization.
            Default 2 percentile
        vmax (float): maximum value for color normalization.
            Default 98 percentile

    Attributes:
        shape (tuple): shape of raster array
        vmin (float): minimum value used for colorscale. Default 2% percentile
        vmax (float): maximum value used for colorscale. Default 98% percentile

    """

    def __init__(self, name, array, transform, projection, **kwargs):
        self.array = array
        self.name = name
        self.transform = transform
        self.projection = projection
        self.shape = array.shape
        self.vmin = kwargs.get("vmin", np.nanpercentile(self.values, 2))
        self.vmax = kwargs.get("vmax", np.nanpercentile(self.values, 98))

    def __repr__(self):
        txt = f"Band {self.name} {self.shape} {self.dtype}\n"
        txt += (
            f"Min:{self.min:g} Max:{self.max:g} Vmin:{self.vmin:g} Vmax:{self.vmax:g}"
        )
        return txt

    def copy(self, **kwargs):
        """Create copy of band

        Transform and projection is not changed.

        Keyword Args:
            name (str): New name. Default is original one
            array (numpy.ma.array): New data. De3fault is original one

        """
        name = kwargs.get("name", self.name)
        array = kwargs.get("array", self.array).copy()
        assert (
            array.shape == self.array.shape
        ), f"Shape of array {array.shape} must be same as original {self.array.shape}"
        return Band(name, array, self.transform, self.projection)

    def setnorm(self, **kwargs):
        """Set color normalization minimum a maximum

        Keyword Args:
            vmin (float): minimum value for color normalization.
                Default 2 percentile
            vmax (float): maximum value for color normalization.
                Default 98 percentile

        """
        self.vmin = kwargs.get("vmin", np.nanpercentile(self.values, 2))
        self.vmax = kwargs.get("vmax", np.nanpercentile(self.values, 98))

    @property
    def values(self):
        """numpy.array: Return 1D array of non-masked values from band"""
        return self.array[np.invert(self.array.mask)].data

    def __array__(self, dtype=None):
        if dtype is None:
            return self.array
        else:
            return self.array.astype(dtype)

    def __add__(self, other):
        if isinstance(other, Band):
            return self.copy(
                array=self.array + other.array, name=f"{self.name}+{other.name}"
            )
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array + other, name=f"{self.name}+{other}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __radd__(self, other):
        if isinstance(other, Band):
            return self.copy(
                array=self.array + other.array, name=f"{other.name}+{self.name}"
            )
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array + other, name=f"{other}+{self.name}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __sub__(self, other):
        if isinstance(other, Band):
            return self.copy(
                array=self.array - other.array, name=f"{self.name}-{other.name}"
            )
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array - other, name=f"{self.name}-{other}")
        else:
            raise ValueError(f"{type(other)} not suppoorted for addition")

    def __rsub__(self, other):
        if isinstance(other, Band):
            return self.copy(
                array=other.array - self.array, name=f"{other.name}-{self.name}"
            )
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
        """numpy.dtype: type of data in band"""
        return self.array.dtype

    def astype(self, dtype):
        """Convert band to other dtype

        Args:
            dtype: numpy dtype

        Returns:
            `s2lx.Band` raster band

        """
        return self.copy(array=self.array.astype(dtype))

    @property
    def min(self):
        """Returns minimum of data"""
        return self.array.min()

    @property
    def max(self):
        """Returns maximum of data"""
        return self.array.max()

    @property
    def norm(self):
        """Returns matplotlib.colors.Normalize using vmin, vmax properties"""
        return colors.Normalize(vmin=self.vmin, vmax=self.vmax, clip=True)

    @property
    def normalized(self):
        """Returns normalized raster values as numpy.array"""
        return self.norm(self.array)

    def apply(self, fun):
        """Apply function to raster data

        You can use pre-defined filters from `s2lx.s2filters`, but any other
        function accepting 2D numpy array could be used.

        Args:
            fun: function

        Returns:
            `s2lx.Band` raster band with new values

        """
        sname = f"{fun.__name__}" + f"({self.name})"
        fvals = np.ma.masked_array(fun(self.array.data), mask=self.array.mask)
        return self.copy(array=fvals, name=sname)

    def patch(self, other, fit=False):
        """Patch bands

        All masked data are patched from other band. Both bands must have
        same transform and projection.

        Args:
            other: `s2lx.Band` raster

        Keyword Args:
            fit (bool): If True, the patching dataset bands are linearly
                scaled to fit in overlapping region. Default False

        Returns:
            `s2lx.Band` patched raster band

        """
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
        """Intersect bands

        Returns tuple of two bands with all values masked except
        intersecting region. Both bands must have same transform
        and projection.

        Args:
            other: `s2lx.Band` raster

        Returns:
            tuple of two `s2lx.Band` raster bands

        """
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
        """Show band

        Create matplotlib figure with raster band data and colorbar.

        Keyword Args:
            figsize (tuple): figure size. Default is matplotlib defaults
            filename(str): if not None, the plot is save to file.
                Default None
            dpi (int): DPI for save image. Default 300
            cmap: matplotlib color map. Default 'cividis'
            under (color): color used for values under vmin. Default cmap(0)
            over (color): color used for values above vmax. Default cmap(1)
            masked (color): color used for masked values. Default 'white'

        """
        figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        showproj = kwargs.get("showproj", True)
        cmap = plt.get_cmap(kwargs.get("cmap", "cividis"))
        cmap.set_under(kwargs.get("under", cmap(0.0)))
        cmap.set_over(kwargs.get("over", cmap(1.0)))
        cmap.set_bad(kwargs.get("masked", "white"))
        # plot
        if "ax" in kwargs:
            ax = kwargs.get("ax")
        else:
            f, ax = plt.subplots(figsize=figsize)
        extent = (
            self.transform[0],
            self.transform[0] + self.shape[1] * self.transform[1],
            self.transform[3] + self.shape[0] * self.transform[5],
            self.transform[3],
        )
        img = ax.imshow(self.array, cmap=cmap, norm=self.norm, extent=extent)
        if showproj:
            ax.set_title(f"{self.name} {pyproj.CRS(self.projection).name}")
        else:
            ax.set_title(f"{self.name}")
        ax.set_aspect(1)
        if "ax" not in kwargs:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            f.colorbar(img, cax=cax, extend="both")
            filename = kwargs.get("filename", None)
            f.tight_layout()
            if filename is not None:
                f.savefig(filename, dpi=kwargs.get("dpi", 300))
            plt.show()

    def save(self, filename, overviews=False):
        """Save band as GeoTIFF

        Args:
            filename (str): GeoTIFF filename

        Keyword Args:
            overviews (bool): build overviews when True. Default False

        """
        # save
        if filename.lower().endswith(".tif"):
            driver_gtiff = gdal.GetDriverByName("GTiff")
            options = [
                "BIGTIFF=IF_NEEDED",
                "COMPRESS=DEFLATE",
                "PROFILE=GeoTIFF",
                "TILED=YES",
                "NUM_THREADS=ALL_CPUS",
            ]
            ds_create = driver_gtiff.Create(
                filename,
                xsize=self.shape[1],
                ysize=self.shape[0],
                bands=1,
                eType=gdal_array.NumericTypeCodeToGDALTypeCode(self.dtype),
                options=options,
            )
            ds_create.SetProjection(self.projection)
            ds_create.SetGeoTransform(self.transform)
            dt = np.array(self.array.data)
            if np.any(self.array.mask):
                dt[self.array.mask] = 0
                ds_create.GetRasterBand(1).SetNoDataValue(0)
            ds_create.GetRasterBand(1).WriteArray(dt)
            ds_create = None
            # Build pyramids
            if overviews:
                ds = gdal.Open(filename, 1)  # 0 = read-only, 1 = read-write.
                options = ["COMPRESS_OVERVIEW", "DEFLATE"]
                ds.BuildOverviews("NEAREST", [4, 8, 16, 32, 64, 128], options=options)
                ds = None
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
        """Show RGB composite

        Create matplotlib figure with RGB composite.

        Keyword Args:
            figsize (tuple): figure size. Default is matplotlib defaults
            filename(str): if not None, the plot is save to file.
                Default None
            dpi (int): DPI for save image. Default 300

        """
        figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        showproj = kwargs.get("showproj", True)
        rgb = np.dstack((self.r.normalized, self.g.normalized, self.b.normalized))
        # plot
        if "ax" in kwargs:
            ax = kwargs.get("ax")
        else:
            f, ax = plt.subplots(figsize=figsize)
        extent = (
            self.transform[0],
            self.transform[0] + self.shape[1] * self.transform[1],
            self.transform[3] + self.shape[0] * self.transform[5],
            self.transform[3],
        )
        ax.imshow(rgb, extent=extent)
        if showproj:
            ax.set_title(
                f"{self.name} [{self.r.name} {self.g.name} {self.b.name}] {pyproj.CRS(self.projection).name}"
            )
        else:
            ax.set_title(f"{self.name} [{self.r.name} {self.g.name} {self.b.name}]")
        ax.set_aspect(1)
        if "ax" not in kwargs:
            filename = kwargs.get("filename", None)
            f.tight_layout()
            if filename is not None:
                f.savefig(filename, dpi=kwargs.get("dpi", 300))
            plt.show()

    def save(self, filename, overviews=False):
        """Save RGB composite as RGBA GeoTIFF

        Alpha channel is generated from mask of bands.

        Args:
            filename (str): GeoTIFF filename

        Keyword Args:
            overviews (bool): build overviews when True. Default False

        """
        if filename.lower().endswith(".tif"):
            driver_gtiff = gdal.GetDriverByName("GTiff")
            options = [
                "BIGTIFF=IF_NEEDED",
                "COMPRESS=DEFLATE",
                "PROFILE=GeoTIFF",
                "TILED=YES",
                "NUM_THREADS=ALL_CPUS",
                "PHOTOMETRIC=RGB",
            ]
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
            # Build pyramids
            if overviews:
                ds = gdal.Open(filename, 1)  # 0 = read-only, 1 = read-write.
                options = ["COMPRESS_OVERVIEW", "DEFLATE"]
                ds.BuildOverviews("NEAREST", [4, 8, 16, 32, 64, 128], options=options)
                ds = None
        else:
            print("filename must have .tif extension")
