"""
Augment images in a image set.

Takes a command line argument, which is the path to a directory containing images.

This will ignore sub-directories, so has to be the final directory which contains
only images.

Saves augmented images in a separate directory:
    "/exports/eddie/scratch/$USER/chopped_augmented"
"""

import os
import sys
import itertools
from Augmentor import Pipeline


def make_pipeline(imageset_path, output_dir):
    """returns an augmentation pipeline for a given image set"""
    p = Pipeline(imageset_path, output_dir)
    p.random_distortion(probability=0.7, grid_width=4, grid_height=4, magnitude=8)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.zoom(probability=0.3, min_factor=1.1, max_factor=1.4)
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
    return p


def is_test_set(imageset_path):
    return imageset_path.split(os.sep)[-2] == "test"


def get_last_part_of_path(path):
    return os.path.join(*path.split(os.sep)[-3:])


def main():
    IMAGE_DIR = sys.argv[1]
    # check that it's a full (and correct) path to the image directory
    if not IMAGE_DIR.startswith("/") or not os.path.isdir(IMAGE_DIR):
        raise ValueError("need a full path to the image_dir")
    print("{} is a valid directory".format(IMAGE_DIR))
    output_path = "/exports/eddie/scratch/{}/chopped_augmented".format(os.environ["USER"])
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    output_last_part = get_last_part_of_path(IMAGE_DIR)
    print("output last part = {}".format(output_last_part))
    full_output_path = os.path.join(output_path, output_last_part)
    print("saving to {}".format(full_output_path))
    if is_test_set(IMAGE_DIR):
        n_sample = 50000
        print("test set, generating {} images".format(n_sample))
    else:
        n_sample = 500000
        print("training set, generating {} images".format(n_sample))
    pipeline = make_pipeline(IMAGE_DIR, full_output_path)
    pipeline.sample(n_sample)


if __name__ == "__main__":
    main()


