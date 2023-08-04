"""Image Processing Functions.

This module contains functions for image processing, including removing connected regions
from binary images based on region properties.

"""

import numpy as np
import skimage.io as skio
from skimage import measure
from skimage.morphology import binary_erosion
import matplotlib.pyplot as plt
import glob
import time
import os
import pandas as pd




def combine_imgs(img, img2, bkg, cmax=1):

    """Combine three images horizontally into a single composite image.

    Parameters
    ----------
    img : numpy.ndarray
        The first image to be combined.
    img2 : numpy.ndarray
        The second image to be combined.
    bkg : numpy.ndarray
        The background image to be combined.
    cmax : int, optional
        The maximum value for scaling the images. Default is 1.

    Returns
    -------
    numpy.ndarray
        The composite image created by horizontally combining the three input images.

    """
    shape = img.shape
    cimg = np.zeros((shape[0], shape[1] * 3), dtype=np.uint8)
    value = (img, img2, bkg)
    for n in range(3):
        temp = value[n] * 255.0 // cmax
        temp[temp > 255] = 255
        cimg[:, n * shape[1]: (n + 1) * shape[1]] = temp

    cimg[:, shape[1] * 1] = 63
    cimg[:, shape[1] * 2] = 63
    return cimg


def show_images(image):

    """Display the input image with a colorbar.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be displayed.

    Returns
    -------
    None
        This function displays the image on the screen.

    """
    plt.imshow(image)
    plt.colorbar()
    plt.show()


def remove_connected_original(img, count_cutoff=16, background=0):

    """Remove connected regions from the input binary image using count-based criteria.

    Parameters
    ----------
    img : numpy.ndarray
        The input binary image.
    count_cutoff : int, optional
        The minimum count of pixels for a connected region to be removed. Default is 16.
    background : int, optional
        The value of the background in the binary image. Default is 0.

    Returns
    -------
    numpy.ndarray
        The cleaned binary image with connected regions removed.
    numpy.ndarray
        The background image containing the removed regions.

    """
    all_labels = measure.label(img > 0, background=background)
    count = np.bincount(all_labels.flatten())
    mask = count > count_cutoff

    img2 = np.copy(img)
    for n in range(1, len(mask)):
        if mask[n]:
            roi = all_labels == n
            val = img[roi]
            if np.min(val) == np.max(val):
                img2[roi] = 0
    bkg = img - img2
    return img2, bkg


def remove_connected_dilation(img, count_cutoff=1, background=0):

    """Remove connected regions from the input binary image using binary erosion.

    Parameters
    ----------
    img : numpy.ndarray
        The input binary image.
    count_cutoff : int, optional
        The minimum count of pixels for a connected region to be removed. Default is 1.
    background : int, optional
        The value of the background in the binary image. Default is 0.

    Returns
    -------
    numpy.ndarray
        The cleaned binary image with connected regions removed.
    numpy.ndarray
        The background image containing the removed regions.

    """
    mask_valid = img > 0
    all_labels = measure.label(mask_valid, background=background)

    mask_valid = binary_erosion(mask_valid)
    count = np.bincount((all_labels * mask_valid).flatten())
    mask = count < count_cutoff

    img2 = np.copy(img)
    for n in range(1, len(mask)):
        if mask[n]:
            roi = all_labels == n
            img2[roi] = 0
    bkg = img - img2
    return img2, bkg


def remove_connected(img, count_cutoff=1, background=0):

    """Remove connected regions from the input binary image based on region properties.

    This function removes connected regions from the input binary image 'img' based on region properties such as area and solidity.
    It uses Scikit-Image's 'measure.label' function to label connected components in the binary image.
    Region properties like area and solidity are calculated using 'measure.regionprops_table'.

    Parameters
    ----------
    img : numpy.ndarray
        The input binary image.
    count_cutoff : int, optional
        The minimum count of pixels for a connected region to be removed. Default is 1.
    background : int, optional
        The value of the background in the binary image. Default is 0.

    Returns
    -------
    numpy.ndarray
        The cleaned binary image with connected regions removed.
    numpy.ndarray
        The background image containing the removed regions.

    """
    mask_valid = img > 0
    all_labels = measure.label(mask_valid, background=background)

    props = measure.regionprops_table(all_labels, properties=('label', 'solidity', 'perimeter', 'area'))
    data = pd.DataFrame(props)
    data = data[data.area >= 5]
    data = data[data.solidity < 0.4]
    data['ratio'] = data.perimeter ** 2 / data.area

    img2 = np.copy(img)
    for index, row in data.iterrows():
        roi = all_labels == row['label']
        img2[roi] = 0

    bkg = img - img2
    return img2, bkg



def show_result(img, img2, bkg):

    """Display the original image, cleaned image, and background image in separate subplots.

    Parameters
    ----------
    img : numpy.ndarray
        The original image.
    img2 : numpy.ndarray
        The cleaned image with connected regions removed.
    bkg : numpy.ndarray
        The background image containing the removed regions.

    Returns
    -------
    None
        This function displays the images on the screen.

    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img, vmin=0, vmax=1)
    ax[0].set_title('raw_data')
    ax[1].imshow(img2, vmin=0, vmax=1)
    ax[1].set_title('clean')
    ax[2].imshow(bkg, vmin=0, vmax=1)
    ax[2].set_title('bkg')
    plt.show()
    plt.close(fig)


    if __name__ == '__main__':
        files = glob.glob('./Raw_Data/*.tif')
        t0 = time.perf_counter()
        for n in range(len(files)):
            print(files[n])
            a = skio.imread(files[n])
            img2, bkg = remove_connected(a)
            cimg = combine_imgs(a, img2, bkg)
            skio.imsave(os.path.basename(files[n]).replace('.tif', '.jpg'), cimg)
        t1 = time.perf_counter()
        print(t1 - t0)