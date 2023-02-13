import numpy as np
from scipy import ndimage
from scipy import fftpack
from skimage.morphology import disk
from skimage.filters import rank

__all__ = ['median_filter', 'gaussian_filter', 'gaussian_laplace_filter',
           'laplace_filter', 'sobel_filter', 'prewitt_filter',
           'majority_filter', 'fft_filter']

def median_filter(**kwargs):
    radius = kwargs.pop('radius', 3)
    kwargs['footprint'] = disk(radius)
    def median(x):
        return ndimage.median_filter(x, **kwargs)
    return median

def gaussian_filter(sigma, **kwargs):
    def gaussian(x):
        return ndimage.gaussian_filter(x, sigma, **kwargs)
    return gaussian

def gaussian_laplace_filter(sigma, **kwargs):
    def gaussian_laplace(x):
        return ndimage.gaussian_laplace(x, sigma, **kwargs)
    return gaussian_laplace

def laplace_filter(**kwargs):
    def laplace(x):
        return ndimage.laplace(x)
    return laplace

def sobel_filter(**kwargs):
    def sobel(x):
        return ndimage.sobel(x)
    return sobel

def prewitt_filter(**kwargs):
    def prewitt(x):
        return ndimage.prewitt(x)
    return prewitt

def majority_filter(**kwargs):
    radius = kwargs.pop('radius', 3)
    kwargs['footprint'] = disk(radius)
    def majority(x):
        return rank.majority(x, **kwargs)
    return majority

def fft_filter(**kwargs):
    fraction = kwargs.pop('fraction', 0.3)
    def fft(x):
        x_fft = fftpack.fft2(x)
        r, c = x_fft.shape
        x_fft[int(r * fraction) : int(r * (1 - fraction))] = 0
        x_fft[:, int(c * fraction) : int(c * (1 - fraction))] = 0
        return fftpack.ifft2(x_fft).real
    return fft