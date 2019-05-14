
from util.visualization.cumstom_confusion_matrix import make_heatmap
import os
import numpy as np
import cv2

def tensor_to_image(image):
    """
    Tries to reshape, convert and do operations necessary to bring the image
    in a format friendly to be saved and logged to Tensorboard by
    save_image_and_log_to_tensorboard()

    Parameters
    ----------
    image : ?
        Image to be converted

    Returns
    -------
    image : ndarray [W x H x C]
        Image, as format friendly to be saved and logged to Tensorboard.

    """
    # Check if the data is still a Variable()
    if 'variable' in str(type(image)):
        image = image.data

    # Check if the data is still on CUDA
    if 'cuda' in str(type(image)):
        image = image.cpu()

    # Check if the data is still on a Tensor
    if 'Tensor' in str(type(image)):
        image = image.numpy()
    assert ('ndarray' in str(type(image)))  # Its an ndarray

    # Check that it does not have anymore the 4th dimension (from the mini-batch)
    if len(image.shape) > 3:
        assert (len(image.shape) == 4)
        image = np.squeeze(image)
    assert (len(image.shape) == 3)  # 3D matrix (W x H x C)

    # Check that the last channel is of size 3 for RGB
    if image.shape[2] != 3:
        assert (image.shape[0] == 3)
        image = np.transpose(image, (1, 2, 0))
    assert (image.shape[2] == 3)  # Last channel is of size 3 for RGB

    # Check that the range is [0:255]
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 255
    assert (image.min() >= 0)  # Data should be in range [0:255]

    return image


confusion_matrix_heatmap = make_heatmap([[98, 23], [28, 152]], ("asbestos  ", "non-asbestos  "))

output_folder = os.path.dirname("/Users/tomasz/")
dest_filename = os.path.join(output_folder, 'images.png')

if not os.path.exists(os.path.dirname(dest_filename)):
    os.makedirs(os.path.dirname(dest_filename))

# Ensuring the data passed as parameter is healthy
image = tensor_to_image(confusion_matrix_heatmap)

# Write image to output folder
cv2.imwrite(dest_filename, image)