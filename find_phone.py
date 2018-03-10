import numpy as np
import argparse
import os
import utils
from PIL import Image


##################################################
# Build the argument parser                      #
##################################################
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', dest='path',
                    action='store',
                    required=True,
                    help='''
                        The full path to the testing image 
                        ''')
args = parser.parse_args()

##################################################
# Predicting                                     #
##################################################

# Load test image informations
images_path = []
images_name = []
images_ndim = []
images_name.append(args.path.split('/')[-1])
images_path.append(args.path)
images_ndim.append(utils.cv2.imread(args.path).shape)
d = {'Image_Name': images_name,
     'Image_Path': images_path,
     'Image_ndim': images_ndim}
test_image = utils.pd.DataFrame(data=d)

# Create pixel to coordinates dictionary
x_ticks, y_ticks = utils.pixel_to_coordinate(test_image)


# Check images are the same dimension used to traing the
# model
row, col, chn = test_image['Image_ndim'].iloc[0]
if chn != 3:
    print("""
            The testing image channels does not match the
            channels used in the training process
        """)
    quit()

# Check that the model was trained by checking for the bottlenecks
if os.path.isfile('unet_trained/checkpoint') and os.stat('unet_trained/checkpoint')[6] > 0:
    print("Predicting...")
    # Build model
    net = utils.unet.Unet(channels=chn, n_class=2, layers=3, features_root=16)
    # Load test image
    img = np.array(Image.open(test_image['Image_Path'].iloc[0]), np.float32)
    # Reshpae image into tensor
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    # Make the prediction
    #prediction = net.predict("/home/gaure/Google_Drive/Machine_Learning/BrainCorp/unet_trained/model.cpkt", img)
    prediction = net.predict("unet_trained/model.cpkt", img)
    # Get the mask pixels that are part of the detected phone
    phone_pixels = prediction[0,...,0] > 0.9
    # Find the index of the pixels in the middle of the predicted phone pixels
    if len(np.transpose(np.nonzero(phone_pixels))) == 0:
           print("No phone was detected")
    else:
        r, c = np.transpose(np.nonzero(phone_pixels))[int(len(np.transpose(np.nonzero(phone_pixels)))/2)]
        #Convert the pixels to coordinate
        for k in x_ticks.keys():
            if x_ticks[k] == c:
                x_tick = k
        for k in y_ticks.keys():
            if y_ticks[k] == r:
                y_tick = k
        # Print the cordinates
        print("{}, {}".format(x_tick, y_tick))
else:
    print("Please run the find_phone_finder.py command to train the model")

