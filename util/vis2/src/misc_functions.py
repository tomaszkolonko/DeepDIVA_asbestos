"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import sys
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map

import torch
from torch.autograd import Variable
import models


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    path_to_output_files_on_server = '/home/thomas.kolonko/generated/viz/vanilla'
    if not os.path.exists(path_to_output_files_on_server):
        os.makedirs(path_to_output_files_on_server)
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join(path_to_output_files_on_server, file_name + '.jpg')
    save_image(gradient, path_to_file)


def save_class_activation_images(org_img, activation_map, file_name, target_layer):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    path_to_output_files_on_server = '/home/thomas.kolonko/generated/vgg13_gradcam_g_16'
    if not os.path.exists(path_to_output_files_on_server):
        os.makedirs(path_to_output_files_on_server)
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join(path_to_output_files_on_server, file_name+'_tl'+ str(target_layer) +'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join(path_to_output_files_on_server, file_name+'_tl'+ str(target_layer) +'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # Save grayscale heatmap
    path_to_file = os.path.join(path_to_output_files_on_server, file_name+'_tl'+ str(target_layer) +'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def save_image(im, path):
    """
        Saves a numpy matrix of shape D(1 or 3) x W x H as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, np.ndarray):
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=0)
        if im.shape[0] == 1:
            # Converting an image with depth = 1 to depth = 3, repeating the same values
            # For some reason PIL complains when I want to save channel image as jpg without
            # additional format in the .save()
            im = np.repeat(im, 3, axis=0)
            # Convert to values to range 1-255 and W,H, D
        if im.shape[0] == 3:
            im = im.transpose(1, 2, 0) * 255
        im = Image.fromarray(im.astype(np.uint8))
    im.save(path)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((224, 224))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = (('/home/thomas.kolonko/DeepDIVA_asbestos/util/vis2/input_images/asbestos-042_7.png', 1),  #asbestos
                    ('/home/thomas.kolonko/DeepDIVA_asbestos/util/vis2/input_images/asbestos-113_2.png', 1),  #asbestos
                    ('/home/thomas.kolonko/DeepDIVA_asbestos/util/vis2/input_images/asbestos-113_8.png', 1),  # asbestos
                    ('/home/thomas.kolonko/DeepDIVA_asbestos/util/vis2/input_images/snap_120968_5_10.png', 1),  #asbestos
                    ('/home/thomas.kolonko/DeepDIVA_asbestos/util/vis2/input_images/snap_120968_17_12.png', 1),  #non-asbestos
                    ('/home/thomas.kolonko/DeepDIVA_asbestos/util/vis2/input_images/snap_120968_7_11.png', 1)) #asbestos
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    file_name_to_export = file_name_to_export + '-' + str(target_class)
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    # Process image
    prep_img = preprocess_image(original_image, resize_im=True)
    # prep_img.save('/home/thomas.kolonko/DeepDIVA_asbestos/util/vis2/generated/yolo/original_image_two.png')
    # Define model

    def load_model_from_file():
        # path_to_checkpoint = '/home/thomas.kolonko/f_vgg13_g_16_optimized/vgg13_bn_g_16_optimized/FINAL/model_name=vgg13_bn_g/epochs=50/lr=0.1/decay_lr=20/momentum=0.499036/weight_decay=1e-05/13-04-19-00h-11m-17s/checkpoint.pth.tar'

        # VGG13_bn_pre for chapter 5
        path_to_checkpoint = '/home/thomas.kolonko/OutputHistory/SIGOPT/output_asbestos_vgg13_bn_sigopt/' \
                             'tz_asbestos_vgg13_bn_sigopt_pre/FINAL/model_name=vgg13_bn/epochs=50/pretrained=True/' \
                             'lr=0.09353319065678362/decay_lr=20/momentum=0.04107414719444524/' \
                             'weight_decay=0.009733960128499166/26-03-19-07h-13m-43s/checkpoint.pth.tar'
        # VGG13_bn for chapter 5
        # path_to_checkpoint = '/home/thomas.kolonko/OutputHistory/SIGOPT/output_asbestos_vgg13_bn_sigopt/' \
        #                      'tz_asbestos_vgg13_bn_sigopt/FINAL/model_name=vgg13_bn/epochs=50/lr=0.05417303921420196/' \
        #                      'decay_lr=20/momentum=0.6435035228001551/weight_decay=0.0032227545216798603/' \
        #                      '28-03-19-22h-26m-56s/checkpoint.pth.tar'

        path_to_checkpoint = '/home/thomas.kolonko/f_vgg13_g_16_optimized/vgg13_bn_g_16_optimized/FINAL/' \
                             'model_name=vgg13_bn_g/epochs=50/lr=0.1/decay_lr=20/momentum=0.499036/weight_decay=1e-05/' \
                             '13-04-19-00h-11m-17s/checkpoint.pth.tar'

        model = models.__dict__['vgg13_bn_g'](output_channels=2, pretrained=False)
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
        return model


    # pretrained_model = models.alexnet(pretrained=True)
    pretrained_model = load_model_from_file()
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)
