# This imports numpy package and declares it as np to be used in the code
import numpy as np
import skimage.io as skio
# This imports the specific measure module from the skimage package
from skimage import measure
from skimage import filters
from skimage.morphology import binary_erosion
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import os
import pandas as pd


# This is the module that looks like its used to combind images it looks like img , img2, and bkg are what will combind
def combine_imgs(img, img2, bkg, cmax=1):
    # shape is a module within numpy used to determin an arrays dimensions
    # in this case its seeing the dimesions of the first imag
    shape = img.shape
    # this is using numpy zeros in order to
    cimg = np.zeros((shape[0], shape[1] * 3), dtype=np.uint8) 
    value = (img, img2, bkg)
    for n in range(3):
        temp = value[n] * 255.0 // cmax
        temp[temp > 255] = 255
        cimg[:, n * shape[1]: (n + 1) * shape[1]] = temp

    # add spacer
    cimg[:, shape[1] * 1] = 63
    cimg[:, shape[1] * 2] = 63
    return cimg

def show_images(image):
    plt.imshow(image, vmax=2)
    plt.colorbar()
    plt.show()
    



# understand every step of this function if its a function 
# what kind of data input is it/ data type
# what is the output and output type

# Youve set the default values for count_cutoff to 16 and for background to 0
def remove_connected_original(img, count_cutoff=16, background=0):
    # Displays the Original image 
    show_images(img)
    # Displays a binary image where pixels greater than 0 are set to True.
    show_images(img > 0)
    # Labels connected components in the binary image.
    all_labels = measure.label(img > 0, background=background)
    # Counts the occurrences of each label.
    count = np.bincount(all_labels.flatten())
    # Creates a boolean mask for labels exceeding the count cutoff.
    mask = count > count_cutoff

    # Creates a copy of the original image.
    img2 = np.copy(img)
    # Iterates over the labels, excluding the background.
    for n in range(1, len(mask)):
        # Checks if the current label satisfies the mask condition.
        if mask[n]:
            # Creates a binary image for the current label.
            roi = all_labels == n
            # Extracts pixel values from the original image within the region of interest.
            val = img[roi]
            # Checks if all pixels within the region have the same value.
            if np.min(val) == np.max(val):
                # Sets pixels in the copied image belonging to the region to 0.
                img2[roi] = 0
    #Creates a new image containing the removed regions.
    bkg = img - img2
    # Returns the modified image (img2) and the background image (bkg)
    return img2, bkg


def remove_connected_dilation(img, count_cutoff=1, background=0):
    mask_valid = img > 0
    all_labels = measure.label(mask_valid, background=background)
    mask_valid = binary_erosion(mask_valid)

    count = np.bincount((all_labels * mask_valid).flatten())
    # mask = count > count_cutoff
    mask = count < count_cutoff

    img2 = np.copy(img)
    for n in range(1, len(mask)):
        if mask[n]:
            roi = all_labels == n
            img2[roi] = 0
    bkg = img - img2
    
    return img2, bkg


def remove_connected(img, count_cutoff=1, background=0):
    mask_valid = img > 0
    all_labels = measure.label(mask_valid, background=background)
    # mask_valid = binary_erosion(mask_valid)

    # count = np.bincount((all_labels * mask_valid).flatten())
    props = measure.regionprops_table(all_labels, properties=('label', 'solidity', 'perimeter', 'area'))

    data = pd.DataFrame(props)
    data = data[data.area >= 5]
    data = data[data.solidity < 0.4]
    data['ratio'] = data.perimeter ** 2 / data.area
    print(data)

    img2 = np.copy(img)
    for index, row in data.iterrows():
        roi = all_labels == row['label']
        img2[roi] = 0
    bkg = img - img2

    return img, bkg


def show_result(img, img2, bkg):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    # imshow is from matplot lab and its purpose is to display images on specified subplots
    ax[0].imshow(img, vmin=0, vmax=1)	
    ax[0].set_title('raw_data')
    ax[1].imshow(img2, vmin=0, vmax=1)	
    ax[1].set_title('clean')

    ax[2].imshow(bkg, vmin=0, vmax=1)	
    ax[2].set_title('bkg')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # The line files = glob.glob('./Argonne-Internship/*.tif') is using the glob module to find all files
    #  with the extension .tif in the Argonne-Internship directory. 
    # The glob.glob() function returns a list of file paths matching the specified pattern.
    files = glob.glob('./Raw_Data/*.tif')
    print(files)
    # this is used to measure the starting time of the code 
    t0 = time.perf_counter()
    # for n in range(len(files)):
    for n in range(1):
        print(files[n])
        # imread is from skimage #
        # The line a = skio.imread(files[n]) reads an image file from the specified file path 
        # and stores the image data in the variable a
        a = skio.imread(files[n])
        # This line calls remove_connected_original and passes the image a as an argument
        img2, bkg = remove_connected_original(a)
        # This line calls combine_imgs and passes three arguments which are a, img2, and bkg
        cimg = combine_imgs(a, img2, bkg)
        # This line saves the composite image cimg as a JPEG file. 
        # The skio.imsave() function from the scikit-image (skimage) library is used for this purpose.
        skio.imsave(os.path.basename(files[n]).replace('.tif', '.jpg'), cimg)
    # this would be the end time of running the code
    t1 = time.perf_counter()
    # this will print the time it took to run the code
    print(t1 - t0)
