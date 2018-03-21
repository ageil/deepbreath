import SimpleITK as sitk
import skimage as ski
import numpy as np
import os, re, glob


def read_image_dir(imdir):
    """Read a directory of images. Images are expected to be named like
    vol([0-9]+)_([0-9]{8}).*

    Return a tuple containing 5-dimensional array of images and an array of paths
    The first dimension is organized by volume
    The second dimension is organized by date (should have size 5)
    The third, fourth, fifth and sixth dimensions holds the image data. 
    The sixth dimension is of size 1 and represents the number of channels, which is required by keras
    """
    paths = sorted(glob.glob(os.path.join(imdir, '*')))
    matcher = re.compile('.*vol([0-9]+)_([0-9]{8}).*')
    used_paths = []
    image_sets = []
    keys = {}
    n = 0
    target_shape = [0,0,0,1]
    for path in paths:
        match = matcher.match(path)
        if match is not None:
            vol = match.group(1)
            sitk_image = sitk.ReadImage(path, sitk.sitkFloat32)
            image = sitk.GetArrayFromImage( sitk_image )

            ## Keep track of supremum shape of image seen so far
            for i in range(3):
                if image.shape[i] > target_shape[i]:
                    target_shape[i] = image.shape[i]
            
            if vol not in keys:
                keys[vol] = n
                n += 1
                image_sets.append([])
                used_paths.append([])

            image_sets[ keys[vol] ].append( image )
            used_paths[ keys[vol] ].append( path )

    ## If needed set target shape to a specific size
    ## target_shape = (128,128,128)

    ## Crop and pad all images to supremum shape and add a fourth dimension with size 1 to satisfy keras
    for i in range(len(image_sets)):
        for j in range(len(image_sets[i])):
            image = image_sets[i][j]
            shape = image.shape

            ## Crop if image is too big
            if shape[0] > target_shape[0]:
                crop = shape[0] - target_shape[0]
                image = image[crop//2:target_shape[0],:]            
            if shape[1] > target_shape[1]:
                crop = shape[1] - target_shape[1]
                image = image[:,crop//2:target_shape[1]]
            if shape[2] > target_shape[2]:
                crop = shape[2] - target_shape[2]
                image = image[:,crop//2:target_shape[2]]

            ## Zero pad if image is too small
            pad0 = target_shape[0] - image.shape[0]
            pad1 = target_shape[1] - image.shape[1]
            pad2 = target_shape[2] - image.shape[2]
            padding = ((pad0//2, pad0//2 + pad0%2), (pad1//2, pad1//2 + pad1%2), (pad2//2, pad2//2 + pad2%2))
            image = ski.util.pad(image, padding, mode='constant')
            image_sets[i][j] = image.reshape(target_shape)

        ## Stack the image set into an array
        image_sets[i] = np.stack(image_sets[i])

    ## Stack all image sets into an array
    image_sets = np.stack( image_sets )

    return image_sets, np.array(used_paths, dtype=np.string_)


