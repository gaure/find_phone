import pandas as pd
import numpy as np
import glob, os
import pickle
import cv2
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import math
from tf_unet import unet, util, image_util
import re
import argparse


##################################################
# Build and image indexing structure             #
##################################################

def load_images(path):
    """

    :param path: Absolute path to a folder that contains the images
    :return: training_images: A pandas dataframe with the images information

    """
    # Create lists that will received the image information
    images_path = []
    images_name = []
    images_ndim = []

    # Check if the pickle file exists to load the image info from it
    if not os.path.isfile('training_images.pckl') or os.stat('training_images.pckl')[6] == 0:

        print("Data has not been loaded, loading")

        # Create file
        file = open('training_images.pckl', 'w').close()

        # Exclude mask
        p = re.compile('.*mask*.|.*label*.')

        # Create dataframe
        for pth in glob.glob(path + '/*'):
            if not p.match(pth):
                images_path.append(pth)
                images_name.append(pth.split('/')[-1])
                images_ndim.append(cv2.imread(pth).shape)

        # Create datafram source dict
        d = {'Image_Name': images_name,
             'Image_Path': images_path,
             'Image_ndim': images_ndim}

        # Create dataframe to index images
        training_images = pd.DataFrame(data=d)

        # Save the dataframe in a pickle file
        with open('training_images.pckl', 'wb') as fh:
            pickle.dump(training_images, fh)
    else:

        print("Data will be loaded from the pickel file")

        # Load dataframe from the pickle file
        training_images = pickle.load(open('training_images.pckl', 'rb'))

    # Add additional dataframe columns to get labels
    training_images['x'] = np.zeros(len(training_images))
    training_images['y'] = np.zeros(len(training_images))

    # Load images labels if exists
    if os.path.isfile(path + '/labels.txt'):
        with open(path + '/labels.txt') as f:
            labels = f.readlines()
        labels = [x.strip() for x in labels]
        # Merge
        for l in labels:
            i, x, y = l.split()
            training_images.loc[training_images.Image_Name == i, ['x', 'y']] = str(x), str(y)
    else:
        print("File label was not provided, please provide it")



    return training_images


##################################################
# Image preprocessing function                   #
##################################################
def get_image(img_path):
    """

    :param: img_path:  Path to an image file
    :return: img: greyscale object 2d array

    """
    # loads image
    img = cv2.imread(img_path)
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


##################################################
# Create pixels to coordinates                   #
##################################################
def pixel_to_coordinate(training_images):
    """

    :param: training_images: Pandas dataframe with image path and ndim
    :return: x_ticks_to_range: Dictionary mapping row pixels to coordinates
             y_ticks_to_range: Dictionary mapping colum pixels to coordinates

    """

    # Capture first image dimension
    row, col, chn = training_images['Image_ndim'].iloc[0]

    # Create a numpy 1d array with the number of vertical and
    # horizontal pixels
    x_range = np.arange(0, col + 1, dtype=float).reshape(-1, 1)
    y_range = np.arange(0, row + 1, dtype=float).reshape(-1, 1)

    # Normalize the below ranges using the sklearn scaler function
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    x_scaler.fit(x_range)
    y_scaler.fit(y_range)

    # Round the scaled coordinates to only 4 decimal points like
    # the labels numbers
    x_ticks = list(x_scaler.transform(x_range).reshape(-1, ))
    for i, x in enumerate(x_ticks):
        if len(str(round(x, 4))) < 6:
            x_ticks[i] = str(round(x, 4)).ljust(6, '0')
        else:
            x_ticks[i] = str(round(x, 4))
    y_ticks = list(y_scaler.transform(y_range).reshape(-1, ))
    for i, y in enumerate(y_ticks):
        if len(str(round(y, 4))) < 6:
            y_ticks[i] = str(round(y, 4)).ljust(6, '0')
        else:
            y_ticks[i] = str(round(y, 4))

    # Create a dictionary that maps pixels to coordinares
    x_ticks_to_range = dict(zip(x_ticks, list(x_range.reshape(-1, ))))
    y_ticks_to_range = dict(zip(y_ticks, list(y_range.reshape(-1, ))))

    return x_ticks_to_range, y_ticks_to_range


##################################################
# Get image mask                                 #
##################################################
def get_mask(img, shape_, xticks, yticks, x, y):
    """

    :param img: numpy matrix image representation
           shape_: image shape
           xticks: dictionary mapping col pixels to x coordinates
           yticks: dictionary mapping row pixels to y coordinates
           x: image label x coordinate
           y: image label y coordinate
    :return: maskedImg: Image mask

    """
    mask = np.zeros(shape_, np.uint8)
    cv2.circle(mask, (int(xticks[str(x)])
                      , int(yticks[str(y)])), 20, (255,), -1)

    (T, thresh) = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.bitwise_not(thresh)

    maskedImg = cv2.bitwise_and(thresh2, mask)

    return maskedImg

##################################################
# Generate model mask labels                     #
##################################################
def generate_labels(training_images):
    """

    :param training_images: dataframe with image information

    """
    # Get the pixel to coordinate dictionary
    xticks, yticks = pixel_to_coordinate(training_images)

    for indx in training_images.index:
        # Get image
        img = get_image(training_images['Image_Path'].iloc[indx])
        # Get image label coordinates
        x = training_images['x'].iloc[indx]
        y = training_images['y'].iloc[indx]
        # Get image mask
        mask = get_mask(img, img.shape, xticks, yticks,x,y)
        # Get image name
        image_name = training_images['Image_Name'].iloc[indx].split('.')
        # Get image mask
        mask_name = '/'.join(training_images['Image_Path'].iloc[indx].split('/')[:-1]) + '/' + image_name[0] + '_mask.' + image_name[1]
        # Check if the mask already exists if not create it
        if not os.path.isfile(mask_name):
            cv2.imwrite(mask_name,mask)
