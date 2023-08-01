# **Introduction**

Here at Argonne, we collect x-rays generated by a synchrotron ring using area detectors for various research purposes. However, some high-energy particles from the universe (known as cosmic rays) can penetrate the atmosphere, and even penetrate our lead walls which are used for keeping outside sources from getting in and scrambling up our x-ray data. They usually leave bright tails on the detector which ruins our data. We want to develop software/algorithms to eliminate the signals of cosmic rays. 








# **Background on cosmic rays**


Cosmic rays are high energy particles that originate from the sun, from outside the solar system in our own galaxy and from distant galaxies. Upon reaching earths atmosphere most rays produce second particles and the bulk dont enter the surface. A significant amount of cosmic rays originate from supernova explosions of stars. During the years from 1930 to 1945, a wide variety of investigations confirmed that the primary cosmic rays are mostly protons, and the secondary radiation produced in the atmosphere is primarily electrons, photons and muons. In 1948, observations with nuclear emulsions carried by balloons to near the top of the atmosphere showed that approximately 10% of the primaries are helium nuclei (alpha particles) and 1% are nuclei of heavier elements such as carbon, iron, and lead. There are 3 types of cosmic rays Primary cosmic rays, secondary cosmic rays and cosmic-ray flux. Primary cosmic rays mostly originate from outside the Solar System and sometimes even outside the Milky Way. Primary cosmic rays are composed mainly of protons and alpha particles (99%), with a small amount of heavier nuclei (≈1%) and an extremely minute proportion of positrons and antiprotons.

![cosmic_ray_path](https://github.com/zanekhalidi/2023Argonne-Internship/assets/136121434/ad790e76-a478-4bdd-9323-c54a1e9561a0)
Because cosmic rays carry electric charge, their direction changes as they travel through magnetic fields. By the time the particles reach us, their paths are completely scrambled, as shown by the blue path. We can't trace them back to their sources. Light travels to us straight from their sources, as shown by the purple path. (Credit: NASA's Goddard Space Flight Center)

Secondary rays. When cosmic rays enter the Earth's atmosphere, they collide with atoms and molecules, mainly oxygen and nitrogen. The interaction produces a cascade of lighter particles, a so-called air shower secondary radiation that rains down, including x-rays, protons, alpha particles, pions, muons, electrons, neutrinos, and neutrons. All of the secondary particles produced by the collision continue onward on paths within about one degree of the primary particle's original path. 



# **Our Algorithm**

**A small Summary of what the code does:**

This code was written in Python and created for the purpose of taking the x-ray images that have been corrupted by cosmic rays and filtering out those cosmic ray bits from the image. We implemented certain *packages* within python such as Numpy, matplotlab, skimage, glob, time, pandas and os. Within the code you will also find *functions*. These different functions serve different purposes and some of them are made specifically to filter out these cosmic rays from our x-ray images. 





**A Summary of each function:**

*combine_imgs*
1. Combines three input images (img, img2, bkg) side by side to create a single composite image.
2. The intensity values of each input image are normalized and adjusted to fit within the range of 0 to 255 before combining.
3. The function returns the composite image.

https://github.com/zanekhalidi/2023Argonne-Internship/blob/8609489428c6cfd0d1398d2bf5564feba2d3a997/ANL_Image_Code.py?plain=1#L17-L32

*remove_connected_original*
1. Removes connected regions (connected components) from the input binary image (img) based on their size.
2. It labels the connected regions in the binary image and calculates the size of each region.
3. Connected regions with a size below the specified *count_cutoff*  are considered noise and are removed by setting their pixels to zero.
4. The function returns the cleaned binary image (img2) and the background image (bkg) containing the removed regions.

https://github.com/zanekhalidi/2023Argonne-Internship/blob/8609489428c6cfd0d1398d2bf5564feba2d3a997/ANL_Image_Code.py?plain=1#L47-L84

*remove_connected_dilation*
1. Removes connected regions from the input binary image (img) using a different approach.
2. It labels connected regions in the binary image and counts the pixels in each region.
3. Connected regions with a size below the specified *count_cutoff* are considered noise and are removed by setting their pixels to zero.
4. The function returns the cleaned binary image (img2) and the background image (bkg) containing the removed regions.

https://github.com/zanekhalidi/2023Argonne-Internship/blob/8609489428c6cfd0d1398d2bf5564feba2d3a997/ANL_Image_Code.py?plain=1#L87-L114

*remove_connected*
1. Removes connected regions from the input binary image (img) based on region properties like area and solidity.
2. It labels connected regions in the binary image and computes region properties such as area and solidity.
3. Regions with an area less than 5 or solidity greater than or equal to 0.4 are considered noise and are removed by setting their pixels to zero.
4. The function returns the cleaned binary image (img2) and the background image (bkg) containing the removed regions.

https://github.com/zanekhalidi/2023Argonne-Internship/blob/8609489428c6cfd0d1398d2bf5564feba2d3a997/ANL_Image_Code.py?plain=1#L117-L143

*show_result*
1. Displays three images side by side in a figure: the original image (img), the cleaned image (img2), and the background image (bkg).

https://github.com/zanekhalidi/2023Argonne-Internship/blob/8609489428c6cfd0d1398d2bf5564feba2d3a997/ANL_Image_Code.py?plain=1#L146-L166

*show_images*
1. Displays a single image in a plot with a color bar to visualize its intensity values.

https://github.com/zanekhalidi/2023Argonne-Internship/blob/8609489428c6cfd0d1398d2bf5564feba2d3a997/ANL_Image_Code.py?plain=1#L34-L37

*__main__*
1. The main part of the code reads multiple image files, applies the *remove_connected_original* function to each image, combines the original image, cleaned image, and background image into a composite image, and saves the composite image as a JPEG file. It measures the time taken to process each image file and prints the time taken for each operation.

https://github.com/zanekhalidi/2023Argonne-Internship/blob/8609489428c6cfd0d1398d2bf5564feba2d3a997/ANL_Image_Code.py?plain=1#L169-L194

**An example of the output:**
![29Feb2023_scan113_point049](https://github.com/zanekhalidi/2023Argonne-Internship/assets/136121434/cebecc69-e878-4319-96b7-617b75e149af)

As you can see here the figure on the bottom of the image to the left that looks similar to a broom is present within the image. This is infact a cosmic ray. We then see that within the next two images it is filtered out. 

**Numpy Style Explanation:**



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



