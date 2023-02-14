from scipy import fftpack, ndimage
from skimage.filters import rank
from skimage.morphology import disk

__all__ = [
    "median_filter",
    "gaussian_filter",
    "gaussian_laplace_filter",
    "laplace_filter",
    "sobel_filter",
    "prewitt_filter",
    "majority_filter",
    "fft_filter",
]


def median_filter(**kwargs):
    radius = kwargs.get("radius", 3)

    def median(x):
        return ndimage.median_filter(x, footprint=disk(radius))

    return median


def gaussian_filter(sigma):
    def gaussian(x):
        return ndimage.gaussian_filter(x, sigma)

    return gaussian


def gaussian_laplace_filter(sigma):
    def gaussian_laplace(x):
        return ndimage.gaussian_laplace(x, sigma)

    return gaussian_laplace


def laplace_filter():
    def laplace(x):
        return ndimage.laplace(x)

    return laplace


def sobel_filter():
    def sobel(x):
        return ndimage.sobel(x)

    return sobel


def prewitt_filter():
    def prewitt(x):
        return ndimage.prewitt(x)

    return prewitt


def majority_filter(**kwargs):
    radius = kwargs.get("radius", 3)

    def majority(x):
        return rank.majority(x, footprint=disk(radius))

    return majority


def fft_filter(**kwargs):
    fraction = kwargs.pop("fraction", 0.3)

    def fft(x):
        x_fft = fftpack.fft2(x)
        r, c = x_fft.shape
        x_fft[int(r * fraction) : int(r * (1 - fraction))] = 0
        x_fft[:, int(c * fraction) : int(c * (1 - fraction))] = 0
        return fftpack.ifft2(x_fft).real

    return fft
