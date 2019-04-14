"""
This script calculates the agreement rate of image classification within folder structure
"""

# Utils
import argparse
import os
from PIL import Image

from os import listdir
from os.path import isfile, join


def calculate_agreement_rate(dataset_folder):
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
    print(recursiveFiles)
    d = dict()
    counter = 0
    number_of_annotators = 4


    for infile in recursiveFiles:
        file_name = infile.split('/')[-1][:-4]
        if "not-asbestos" in infile:
            if file_name not in d:
                d[file_name] = -1
            else:
                d[file_name] -= 1
        else:
            if file_name not in d:
                d[file_name] = 1
            else:
                d[file_name] += 1

    for item in d:
        if d[item] is number_of_annotators or d[item] is (-number_of_annotators):
            counter += 1

    print(str(counter / d.__len__()))

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script calculates the agreement rate')

    parser.add_argument('--dataset-folder',
                        help='path to the dataset.',
                        required=True,
                        type=str,
                        default=None)

    args = parser.parse_args()

    calculate_agreement_rate(dataset_folder=args.dataset_folder)