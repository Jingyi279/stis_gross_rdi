from pathlib import Path

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy import units
from astropy.stats import sigma_clipped_stats

def rolling_median(array, window=1):
    # replace each pixel with the median of its four neighbors
    median_filtered = np.zeros(array.size)
    for i in np.arange(array.size):
        lb, ub = i-window, i+window+1
        lb = max([ lb, 0 ])
        ub = min([ ub, array.size ])
        median_filtered[i] = np.nanmedian(array[lb:ub])
    return median_filtered


def median_neighbor_replace(array):
    """
    Median replace each pixel in a 1-D array with the median of its 4 neighbors
    """
    window = 2
    clean_array = np.zeros_like(array)*np.nan
    for i in range(len(array)):
        lb, ub = i-window, i+window+1        
        lb = max([ lb, 0 ]) + 1
        ub = min([ ub, array.size ])
        subset = np.concat((array[lb:i], array[lb+1:ub]))
        clean_array[i] = np.nanmedian(subset)
    return clean_array


def median_filter_image(img, window=5):
    filtered_img = np.zeros_like(img) * np.nan
    for row in np.arange(filtered_img.shape[0]):
        # filtered_img[row] = rolling_median(img[row], window)
        filtered_img[row] = median_neighbor_replace(img[row])
    return filtered_img


def mean_replace(img, coord):
    """
    Mean replace the value at coord with the avg of the neighboring pixels
    """
    # skip values near edges
    # if any([i < 2 for i in coord]): 
    if coord[1] < 2:
        return
    if (coord[0] > img.shape[0]-3) or (coord[1] > img.shape[1]-3):
        return

    box = img[coord[0], coord[1]-2:coord[1]+3]
    mask = np.array(
        [True, True, False, True, True],
    )
    avg = np.nanmean(box[mask])
    print(coord, img[coord[0], coord[1]], avg)
    img[coord[0], coord[1]] = avg
    return


def clean_bad_pixels(
        img : np.ndarray[float],
        # row : int,
        std_thresh : float = 50,
) -> np.ndarray[float]:
    """
    Apply the hot and cold pixel cleaning described in Roberge et al., 2005, Section 3.3
    https://ui.adsabs.harvard.edu/abs/2005ApJ...622.1171R/abstract

    Define a box of size (nrows, img.shape[1]) - i.e. it takes the entire
    dispersion direction, and a few rows in y. For each column, compute the
    local noise as (std_box**2 + std_col-with-outlier-pixel**2)**0.5. If the
    brightest/coldest pixel is more than 5-sigma outside the local noise, it is
    replaced by the median value.

    Parameters
    ----------
    img : np.ndarray
      the 2-D spectral image
    nrows : int = 5
      How many rows to include in the clippnig box.

    Output
    ------
    Define your output

    """
    for row in img:
        avg, med, std = sigma_clipped_stats(row)
        bad_pix = np.where(np.abs(row - avg) > std_thresh*std)[0]
        for bp in bad_pix:
            row[bp] = sigma_clipped_stats(row[bp-50:bp+51])[1]
    return
