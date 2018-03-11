import utils
import argparse
import os

# Supress tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

##################################################
# Build the argument parser                      #
##################################################
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', dest='path',
                    action='store',
                    required=True,
                    help='''
                        The full path to the directory where the 
                        images are located is required
                        ''')
args = parser.parse_args()

#################################################
# Training the find phone model                 #
#################################################
# Load images
training_images = utils.load_images(args.path)
#Generate mask
utils.generate_labels(training_images)
# Create Image data provider
data_provider = utils.image_util.ImageDataProvider(search_path=args.path + '/*',data_suffix='.jpg',mask_suffix='_mask.jpg')
# Create model
net = utils.unet.Unet(channels=data_provider.channels, n_class=data_provider.n_class, layers=3, features_root=16)
# Define model trainer parameters
trainer = utils.unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
# Train model
path = trainer.train(data_provider, "./unet_trained", training_iters=5,  epochs=10, display_step=2)

print("Model was succefully trained")
