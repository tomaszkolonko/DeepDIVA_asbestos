"""
This script allows to crop the asbestos images into 256x256 images for input into alexnet
"""

# Utils
import argparse
import os
from PIL import Image
from template.runner.image_classification_full_image.transform_library import functional


def test_resize_method(dataset_folder):
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

    size = 1024, 1024


    """
    # resampling filters
    NEAREST = NONE = 0
    BOX = 4
    BILINEAR = LINEAR = 2
    HAMMING = 5
    BICUBIC = CUBIC = 3
    LANCZOS = ANTIALIAS = 1
"""
    for image_file in recursiveFiles:
        file_name = image_file[:-4]
        image = Image.open(image_file)
        mode = [0, 1, 2, 3, 4, 5]
        for m in mode:
            image_new = functional.conditional_mirroring(image, m)
            image_new.save(file_name + "_" + str(m) + ".png")

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

    test_resize_method(dataset_folder=args.dataset_folder)
