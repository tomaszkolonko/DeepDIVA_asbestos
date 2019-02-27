"""
This script allows to crop the asbestos images into 256x256 images for input into alexnet
"""

# Utils
import argparse
import os
from PIL import Image
from template.runner.image_classification_random_nine.transform_library import functional



def rotate_images(dataset_folder):
    """
    Resize all images contained within dataset_folder.
    Parameters
    ----------
    dataset_folder : str
        Path to the dataset folder
    Returns
    -------
        None
    """

    # Get all the files recursively from the given dataset_folder
    recursiveFiles = [os.path.join(dp, f) for dp, dn, fn in os.walk(dataset_folder) for f in fn if f.endswith('png')]

    size = 224, 224

    for image_file in recursiveFiles:
        file_name = image_file[:-4]
        image = Image.open(image_file)
        image_arrray = functional.custom_nine_crop(image, size)
        counter = 0
        for img in image_arrray:
            counter += 1
            img.save(file_name + "_" + str(counter) + ".png")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script rotates all images and saves them all')

    parser.add_argument('--dataset-folder',
                        help='path to the dataset.',
                        required=True,
                        type=str,
                        default=None)

    args = parser.parse_args()

    rotate_images(dataset_folder=args.dataset_folder)
