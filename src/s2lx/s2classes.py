import sys
import re
from pathlib import Path
import numbers
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector

from osgeo import gdal
from osgeo import gdal_array
from shapely import wkt, ops, geometry
import pyproj
from sklearn import decomposition
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer


class SAFE:
    """Class to manipulate Sentinel-2 product"""

    def __init__(self, xml):
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
                    xml=xmlfile, crs=crs, desc=desc, bands=bands, meta=meta, transform=sds.GetGeoTransform(), projection=sds.GetProjection()
                )
                sds = None
            ds = None
        else:
            print("Error: No valid filename")

    def __repr__(self):
        txt = f'{self.meta["DATATAKE_1_SPACECRAFT_NAME"]} {self.meta["PRODUCT_TYPE"]} {self.meta["PROCESSING_LEVEL"]}\n'
        txt += f'Sensing date {self.meta["DATATAKE_1_DATATAKE_SENSING_START"]}\n'
        for d in self.datasets:
            txt += (
                f'Dataset {d} bands: {sorted(list(self.datasets[d]["bands"].keys()))}\n'
            )
        return txt

    def preview(self, **kwargs):
        sds = gdal.Open(self.datasets["TCI"]["xml"])
        tci = np.dstack(
            (
                sds.GetRasterBand(
                    self.datasets["TCI"]["bands"]["B4"]["n"]
                ).ReadAsArray(),
                sds.GetRasterBand(
                    self.datasets["TCI"]["bands"]["B3"]["n"]
                ).ReadAsArray(),
                sds.GetRasterBand(
                    self.datasets["TCI"]["bands"]["B2"]["n"]
                ).ReadAsArray(),
            )
        )
        figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        # plot
        f, ax = plt.subplots(figsize=figsize)
        ax.imshow(tci)
        ax.set_title(self.meta['PRODUCT_URI'])
        ax.set_aspect(1)
        f.tight_layout()
        plt.show()

    def get(self, dataset, band):
        if dataset in self.datasets:
            if band in self.datasets[dataset]["bands"]:
                sds = gdal.Open(self.datasets[dataset]["xml"])
                meta = self.datasets[dataset]["bands"][band]
                return sds.GetRasterBand(meta["n"]).ReadAsArray(), meta

    def gclip20(self, name="Clip", band="B12"):
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
            return self.clip20(projWin=(ulx, uly, lrx, lry), name=name)

    def clip20geojson(self, geojson, name="Clip", include8a=False):
        with open(geojson) as f:
            features = json.load(f)["features"]
        bnds = geometry.shape(features[0]["geometry"]).buffer(0).bounds
        return self.clip20(projWin=(bnds[0], bnds[3], bnds[2], bnds[1]), name=name, include8a=include8a)

    def clip20(self, projWin=None, name="Clip", include8a=False):
        if projWin is not None:
            wgs84 = pyproj.CRS('EPSG:4326')
            dstcrs = pyproj.CRS(self.datasets["20m"]["crs"])
            reproject = pyproj.Transformer.from_crs(wgs84, dstcrs, always_xy=True).transform
            # safe footprint
            footprint = ops.transform(reproject, wkt.loads(self.meta['FOOTPRINT'])).buffer(-200)
            # prepare mask
            cg = np.arange(projWin[0] + 10, projWin[2], 20)
            rg = np.arange(projWin[1] - 10, projWin[3], -20)
            xx, yy = np.meshgrid(cg, rg)
            footpath = path.Path(np.array(footprint.exterior.xy).T)
            mask = np.invert(np.reshape(footpath.contains_points(np.array([xx.ravel(), yy.ravel()]).T), xx.shape))
            # go
            meta = self.meta.copy()
            rasters = []
            # 20m
            sds = gdal.Translate(
                "/vsimem/in_memory_output.tif",
                gdal.Open(self.datasets["20m"]["xml"]),
                projWin=projWin,
                xRes=20,
                yRes=20,
                resampleAlg=gdal.GRA_Average,
            )
            for band in ["B5", "B6", "B7", "B8A", "B11", "B12"]:
                bmeta = self.datasets["20m"]["bands"][band].copy()
                rb = sds.GetRasterBand(bmeta.pop("n"))
                vals = rb.ReadAsArray().astype(np.int16)
                if band == "B8A":
                    if include8a:
                        rasters.append(Band('a8', np.ma.masked_array(vals, mask), sds.GetGeoTransform(), sds.GetProjection()))
                        meta['a8'] = bmeta
                else:
                    rasters.append(Band(band.lower(), np.ma.masked_array(vals, mask), sds.GetGeoTransform(), sds.GetProjection()))
                    meta[band.lower()] = bmeta
            sds = gdal.Translate(
                "/vsimem/in_memory_output.tif",
                gdal.Open(self.datasets["10m"]["xml"]),
                projWin=projWin,
                xRes=20,
                yRes=20,
                resampleAlg=gdal.GRA_Average,
            )
            for band in ["B4", "B3", "B2", "B8"]:
                bmeta = self.datasets["10m"]["bands"][band].copy()
                rb = sds.GetRasterBand(bmeta.pop("n"))
                vals = rb.ReadAsArray().astype(np.int16)
                rasters.append(Band(band.lower(), np.ma.masked_array(vals, mask), sds.GetGeoTransform(), sds.GetProjection()))
                meta[band.lower()] = bmeta
            return S2(*rasters, meta=meta, name=name)


class S2:
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
        return sorted([k.name for k in self._bands])

    def restored_pca(self, **kwargs):
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
        meta = dict(components=pca.components_, ev=pca.explained_variance_, evr=pca.explained_variance_ratio_)
        return S2(*rasters, meta=meta, name=self.name + ' PCA restored')

    def pca(self, **kwargs):
        centered = kwargs.get("centered", True)
        n = kwargs.get("n", len(self._bands))
        X = np.array([b.values for b in self._bands]).T
        pca = decomposition.PCA(n_components=n)
        y_pred = pca.fit_transform(X)
        rasters = []
        for ix in range(n):
            vals = self._bands[0].array.copy()
            vals[np.invert(self._bands[0].array.mask)] = y_pred[:, ix]
            rasters.append(self._bands[0].copy(array=vals, name=f'pc{ix}'))
        meta = dict(components=pca.components_, ev=pca.explained_variance_, evr=pca.explained_variance_ratio_)
        return S2(*rasters, meta=meta, name=self.name + ' PCA')


class Band:
    def __init__(self, name, array, transform, projection):
        self.array = array
        self.name = name
        self.transform = transform
        self.projection = projection
        self.shape = array.shape
        self.vmin = np.nanpercentile(self.values, 2)
        self.vmax = np.nanpercentile(self.values, 98)

    def __repr__(self):
        return f'Band {self.name} {self.shape} {self.dtype}\nMin:{self.min:g} Max:{self.max:g} Vmin:{self.vmin:g} Vmax:{self.vmax:g}'

    def copy(self, **kwargs):
        name = kwargs.get('name', self.name)
        array = kwargs.get('array', self.array).copy()
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
            return self.copy(array=self.array + other.array, name=f'{self.name}+{other.name}')
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array + other, name=f'{self.name}+{other}')
        else:
            raise ValueError(f'{type(other)} not suppoorted for addition')

    def __radd__(self, other):
        if isinstance(other, Band):
            return self.copy(array=self.array + other.array, name=f'{other.name}+{self.name}')
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array + other, name=f'{other}+{self.name}')
        else:
            raise ValueError(f'{type(other)} not suppoorted for addition')

    def __sub__(self, other):
        if isinstance(other, Band):
            return self.copy(array=self.array - other.array, name=f'{self.name}-{other.name}')
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array - other, name=f'{self.name}-{other}')
        else:
            raise ValueError(f'{type(other)} not suppoorted for addition')

    def __rsub__(self, other):
        if isinstance(other, Band):
            return self.copy(array=other.array - self.array, name=f'{other.name}-{self.name}')
        elif isinstance(other, numbers.Number):
            return self.copy(array=other - self.array, name=f'{other}-{self.name}')
        else:
            raise ValueError(f'{type(other)} not suppoorted for addition')

    def __mul__(self, other):
        if '+' in self.name or '-' in self.name:
            sname = f'({self.name})'
        else:
            sname = self.name
        if isinstance(other, Band):
            if '+' in other.name or '-' in other.name:
                oname = f'({other.name})'
            else:
                oname = other.name
            return self.copy(array=self.array * other.array, name=f'{sname}*{oname}')
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array * other, name=f'{sname}*{other}')
        else:
            raise ValueError(f'{type(other)} not suppoorted for addition')

    def __rmul__(self, other):
        if '+' in self.name or '-' in self.name:
            sname = f'({self.name})'
        else:
            sname = self.name
        if isinstance(other, Band):
            if '+' in other.name or '-' in other.name:
                oname = f'({other.name})'
            else:
                oname = other.name
            return self.copy(array=self.array * other.array, name=f'{oname}*{sname}')
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array * other, name = f'{other}*{sname}')
        else:
            raise ValueError(f'{type(other)} not suppoorted for addition')

    def __truediv__(self, other):
        if '+' in self.name or '-' in self.name:
            sname = f'({self.name})'
        else:
            sname = self.name
        if isinstance(other, Band):
            if '+' in other.name or '-' in other.name:
                oname = f'({other.name})'
            else:
                oname = other.name
            return self.copy(array=self.array / other.array, name=f'{sname}/{oname}')
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array / other, name=f'{sname}/{other}')
        else:
            raise ValueError(f'{type(other)} not suppoorted for addition')

    def __rtruediv__(self, other):
        if '+' in self.name or '-' in self.name:
            sname = f'({self.name})'
        else:
            sname = self.name
        if isinstance(other, Band):
            if '+' in other.name or '-' in other.name:
                oname = f'({other.name})'
            else:
                oname = other.name
            return self.copy(array=self.array / other.array, name=f'{oname}/{sname}')
        elif isinstance(other, numbers.Number):
            return self.copy(array=self.array / other, name=f'{other}/{sname}')
        else:
            raise ValueError(f'{type(other)} not suppoorted for addition')

    def __abs__(self):
        return Band(f'|{self.name}|', abs(self.array))

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
        sname = f'{fun.__name__}' + f'({self.name})'
        vals = self.array.copy()
        vals[np.invert(self.array.mask)] = fun(self.values)
        return self.copy(array=vals, name=sname)

    def patch(self, other, fit=False):
        assert self.transform == other.transform, 'transform must be identical'
        assert self.projection == other.projection, 'projection must be identical'
        vals = np.array(self.array)
        mask = np.array(self.array.mask)
        ovals = np.array(other.array)[mask]
        if fit:
            o1, o2 = self.intersect(other)
            pp = np.polyfit(o2.values, o1.values, 1)
            ovals = ovals*pp[0] + pp[1]
        vals[mask] = ovals
        mask[mask] = np.array(other.array.mask)[mask]
        return self.copy(array=np.ma.masked_array(vals, mask))

    def intersect(self, other):
        assert self.transform == other.transform, 'transform must be identical'
        assert self.projection == other.projection, 'projection must be identical'
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
        ax.set_title(f'{self.name}')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = f.colorbar(img, cax=cax, extend="both")
        ax.set_aspect(1)
        f.tight_layout()
        plt.show()

    def save(self, filename, **kwargs):
        # save
        if filename.lower().endswith('.tif'):
            driver_gtiff = gdal.GetDriverByName('GTiff')
            ds_create = driver_gtiff.Create(filename, xsize=self.shape[1], ysize=self.shape[0], bands=1,
                                            eType=gdal_array.NumericTypeCodeToGDALTypeCode(self.dtype))
            ds_create.SetProjection(self.projection)
            ds_create.SetGeoTransform(self.transform)
            dt = np.array(self.array.data)
            if np.any(self.array.mask):
                dt[self.array.mask] = 0
                ds_create.GetRasterBand(1).SetNoDataValue(0)
            ds_create.GetRasterBand(1).WriteArray(dt)
            ds_create = None
        else:
            print('filename must have .tif extension')

class Composite:
    def __init__(self, r, g, b, name='RGB composite'):
        assert r.shape == g.shape == b.shape, 'Shapes of bands must be same'
        self.name = name
        self.r = r
        self.g = g
        self.b = b
        self.transform = r.transform
        self.projection = r.projection
        self.shape = r.shape

    def __repr__(self):
        return f'Composite {self.name} {self.shape} [{self.r.name} {self.g.name} {self.b.name}]'

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
        img = ax.imshow(rgb)
        ax.set_title(f'{self.name} [{self.r.name} {self.g.name} {self.b.name}]')
        ax.set_aspect(1)
        f.tight_layout()
        plt.show()

    def save(self, filename, **kwargs):
        # save
        if filename.lower().endswith('.tif'):
            driver_gtiff = gdal.GetDriverByName('GTiff')
            options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
            ds_create = driver_gtiff.Create(filename, xsize=self.shape[1], ysize=self.shape[0],
                                            bands=4, eType=gdal.GDT_Float64, options=options)
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
            print('filename must have .tif extension')

