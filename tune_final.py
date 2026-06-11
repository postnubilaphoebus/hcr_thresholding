"""
Threshold sweep vs. intrinsic (correlation) dimension.

For each intensity threshold T in 1..255, the positive voxels (img > T) are
treated as a point cloud and their Grassberger-Procaccia correlation dimension
is estimated with scikit-dimension's CorrInt. The idea: nonspecific/background
inclusion fills space (D -> ambient dimension, 2 for an image or 3 for a stack),
whereas a real, spatially restricted structure is intrinsically low-dimensional
(D ~ 1-2). The threshold of interest is where D leaves its low plateau and
starts climbing toward the ambient dimension.

Requires: scikit-dimension, scikit-image, tqdm, matplotlib, numpy.
"""

import numpy as np
import skimage.io
import skdim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from skimage.filters import (
  threshold_otsu,
  threshold_yen,
  threshold_li,
  threshold_isodata,
  threshold_triangle,
  threshold_mean,
  threshold_minimum
)
import tifffile
import edt
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
IMAGE_PATH = "T_AVG_kiss1.tif"
BEST_THRESHOLD = 21        # the threshold to mark with a vertical line
CHANNEL = 0                # used only if the image looks multi-channel (RGB/RGBA)

# CorrInt neighbour-rank window = the SCALE the dimension is measured at.
# Larger ranks -> larger radii -> ignores sub-feature noise. See notes below.
K1, K2 = 10, 60

MAX_POINTS = 5000          # subsample the positive set above this (speed + stability)
MIN_POINTS = K2 + 10       # below this, the estimate is meaningless -> NaN
RANDOM_SEED = 0
TARGET_DIR = "/home/laurids/Desktop/hcrs_coped/tuned_results"
# ----------------------------------------------------------------------


def load_uint8(path, channel):
    img = skimage.io.imread(path)
    # Treat a trailing axis of length 3/4 as colour channels, not a spatial axis.
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        print(f"Image looks multi-channel {img.shape}; using channel {channel}.")
        img = img[..., channel]
    # Cast to uint8 (rescale if the data uses a wider range, e.g. 16-bit).
    if img.dtype != np.uint8:
        m = float(img.max())
        img = (img.astype(np.float64) * (255.0 / m)).round().astype(np.uint8) if m > 0 \
            else img.astype(np.uint8)
    print(f"Working with {img.ndim}D image, shape {img.shape}, dtype {img.dtype}.")
    return img


def correlation_dimension(coords):
    rng = np.random.default_rng(RANDOM_SEED)
    num_resample = 3
    n = len(coords)
    if n < MIN_POINTS:
        return np.nan
    mean_val = 0
    # for _ in range(num_resample):
    if n > MAX_POINTS:
        coords_ = coords[rng.choice(n, MAX_POINTS, replace=False)]
    try:
        mean_val += skdim.id.CorrInt(k1=K1, k2=K2).fit(coords_).dimension_
    except Exception:
        return np.nan
    return mean_val
    
import numpy as np
from scipy.optimize import curve_fit
 
 
def exp_model(z, a, b, c):
    """B(z) = a*exp(b*z) + c. The +c lets the curve start at/near 0."""
    return a * np.exp(b * z) + c
 
 
def fit_brightness(z, brightness):
    """
    Fit the exponential to measured per-slice mean brightness.
 
    z, brightness : 1D arrays of equal length (z = slice indices).
    Returns popt = (a, b, c).
    """
    z = np.asarray(z, dtype=float)
    brightness = np.asarray(brightness, dtype=float)
 
    # Data-driven initial guess via a log-linear fit (after removing baseline).
    ymin = brightness.min()
    y = np.clip(brightness - ymin + 1e-3, 1e-3, None)
    slope, intercept = np.polyfit(z, np.log(y), 1)
    p0 = (np.exp(intercept), slope, ymin)
 
    popt, _ = curve_fit(exp_model, z, brightness, p0=p0, maxfev=100000)
    return popt
 
 
def correction_factors(popt, n_slices, n_measured=None,
                       ref="max", max_gain=50.0, eps=1e-6):
    """
    Per-slice multiplicative correction factors.
 
    popt        : (a, b, c) from fit_brightness
    n_slices    : number of output slices (may exceed the measured range)
    n_measured  : slice count used to define the target level (default n_slices).
                  Keep this = your measured stack depth (e.g. 275) so the target
                  doesn't drift when you extend past the data.
    ref         : target brightness to normalize TO.
                  "max" | "mean" | "first" over the measured range, or a float.
    max_gain    : clamp on the factor; prevents near-zero slices blowing up.
    eps         : floor on fitted brightness to avoid divide-by-zero.
 
    Returns (factors, fitted_brightness), each length n_slices.
    """
    if n_measured is None:
        n_measured = n_slices
 
    z = np.arange(n_slices)
    fit = exp_model(z, *popt)
 
    measured = exp_model(np.arange(n_measured), *popt)
    if ref == "max":
        target = measured.max()
    elif ref == "mean":
        target = measured.mean()
    elif ref == "first":
        target = measured[0]
    else:
        target = float(ref)
 
    factors = target / np.maximum(fit, eps)
    return np.clip(factors, 0.0, max_gain), fit


def find_threshold(img, scaling_vals):
    vals = scaling_vals
    img = img / vals[:, None, None]
    img = (img - img.min()) / (img.max() - img.min())
    img = img * 255
    img = img.astype(np.uint8)
    num_above_100 = (img > 100).astype(bool).sum()
    th_otsu = threshold_otsu(img)
    boolean_image = (img >= th_otsu).astype(bool)
    # bool_gaussed = gaussian_filter(boolean_image.astype(np.float32), 10)
    # bool_gaussed_th = threshold_otsu(bool_gaussed)
    # bool_gaussed_final = (bool_gaussed > bool_gaussed_th).astype(bool)
    mask_fraction = boolean_image.sum() / boolean_image.size
    if mask_fraction > 0.1:
        if num_above_100 > 1000:
            #img = img * mask
            thresholds = np.arange(1, 100, 1)
            dims = np.full(thresholds.shape, np.nan)
            for i, T in enumerate(tqdm(thresholds, desc="dimension per threshold")):
                coords = np.argwhere(img > T).astype(np.float64)
                dims[i] = correlation_dimension(coords)
            min_loc = np.argmin(dims)
            BEST_THRESHOLD = thresholds[min_loc]
            ideal_mask = (img > BEST_THRESHOLD).astype(bool)
        else:
            ideal_mask = np.zeros_like(img).astype(bool)
    else:
        ideal_mask = boolean_image
    return ideal_mask

if __name__ == "__main__":
    filenames = os.listdir(os.getcwd())
    filenames = [f for f in filenames if "T_AVG" in f]
    mask = skimage.io.imread("./mask/HCR_skinning_mask_no_eyes.tif").astype(bool)
    edt_mask = edt.edt(mask)
    for fname in tqdm(filenames):
        img = skimage.io.imread(fname)
        #z_profile = np.mean(img.reshape(img.shape[0], -1), axis = 1)
        num_above_100 = (img > 100).astype(bool).sum()
        vals = np.linspace(1, 2, 359)
        img = img / vals[:, None, None]
        img = (img - img.min()) / (img.max() - img.min())
        img = img * 255
        img = img.astype(np.uint8)
        #vals_in_mask = img[mask]
        th_otsu = threshold_otsu(img)
        boolean_image = (img > th_otsu).astype(bool)
        boolean_image = boolean_image * mask
        mask_fraction = boolean_image.sum() / mask.sum()
        if mask_fraction > 0.2:
            if num_above_100 > 1000:
                img = img * mask
                thresholds = np.arange(1, 100, 1)
                dims = np.full(thresholds.shape, np.nan)
                for i, T in enumerate(tqdm(thresholds, desc="dimension per threshold")):
                    coords = np.argwhere(img > T).astype(np.float64)
                    dims[i] = correlation_dimension(coords)
                min_loc = np.argmin(dims)
                BEST_THRESHOLD = thresholds[min_loc]
                ideal_mask = (img > BEST_THRESHOLD).astype(bool)
            else:
                ideal_mask = np.zeros_like(img).astype(bool)
        else:
            ideal_mask = boolean_image

        thirty_distance = (edt_mask > 20).astype(bool)
        ideal_mask = ideal_mask & thirty_distance
        pruned_name = fname.split("AVG_")[1]# T_AVG_sox1b.tif
        tifffile.imwrite(os.path.join(TARGET_DIR, pruned_name), ideal_mask)

        

        