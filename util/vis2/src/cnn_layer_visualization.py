"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import sys
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models
import models

from misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.path_to_output_files = '/home/thomas.kolonko/generated_vgg13_bn'
        # Create the folder to export images if not exists
        if not os.path.exists(self.path_to_output_files):
            os.makedirs(self.path_to_output_files)

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss[0].data.numpy()[0]))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 30 == 0:
                im_path = self.path_to_output_files + '/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss[0].data.numpy()[0]))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 30 == 0:
                im_path = self.path_to_output_files + '/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


def load_model_from_file():
    # VGG13_bn_pre for chapter 5
    path_to_checkpoint = '/home/thomas.kolonko/OutputHistory/SIGOPT/output_asbestos_vgg13_bn_sigopt/' \
                         'tz_asbestos_vgg13_bn_sigopt_pre/FINAL/model_name=vgg13_bn/epochs=50/pretrained=True/' \
                         'lr=0.09353319065678362/decay_lr=20/momentum=0.04107414719444524/' \
                         'weight_decay=0.009733960128499166/26-03-19-07h-13m-43s/checkpoint.pth.tar'
    # VGG13_bn for chapter 5
    path_to_checkpoint = '/home/thomas.kolonko/OutputHistory/SIGOPT/output_asbestos_vgg13_bn_sigopt/' \
                         'tz_asbestos_vgg13_bn_sigopt/FINAL/model_name=vgg13_bn/epochs=50/lr=0.05417303921420196/' \
                         'decay_lr=20/momentum=0.6435035228001551/weight_decay=0.0032227545216798603/' \
                         '28-03-19-22h-26m-56s/checkpoint.pth.tar'

    # path_to_checkpoint = '/home/thomas.kolonko/f_vgg13_g_16_optimized/vgg13_bn_g_16_optimized/FINAL/model_name=vgg13_bn_g/epochs=50/lr=0.1/decay_lr=20/momentum=0.499036/weight_decay=1e-05/13-04-19-00h-11m-17s/checkpoint.pth.tar'
    # path_to_checkpoint = '/home/thomas.kolonko/yolo/tz_asbestos_densenet121_sigopt/FINAL/model_name=densenet121/epochs=50/lr=0.01/decay_lr=20/momentum=0.9/07-04-19-11h-54m-41s/checkpoint.pth.tar'
    model = models.__dict__['vgg13_bn'](output_channels=2, pretrained=False)
    if os.path.isfile(path_to_checkpoint):
        # TODO: Remove or make param: map_location
        model_dict = torch.load(path_to_checkpoint, map_location='cpu')
        print('Loading a saved model')
        try:
            model.load_state_dict(model_dict['state_dict'], strict=False)
        except Exception as exp:
            print(exp)
    else:
        print("couldn't load model from checkpoint")
        sys.exit(-1)
    return model.features

if __name__ == '__main__':
    cnn_layer = [0,3,7,10,14,17,21,24,28,31]
    # filter_pos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    filter_pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # Fully connected layer is not needed
    load_model_from_file()
    pretrained_model_original = models.vgg13_bn(pretrained=False).features
    pretrained_model_new = load_model_from_file()
    for actual_layer in cnn_layer:
        for actual_pos in filter_pos:
            layer_vis = CNNLayerVisualization(pretrained_model_new, actual_layer, actual_pos)
            layer_vis.visualise_layer_with_hooks()
    # Layer visualization with pytorch hooks

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
