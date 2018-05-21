"""
Generate test dataset for a single cell-line.

All data in the 'test' directory.
    - move individual cell-lines data into single directory
    - update filenames of moved data to avoid naming duplicates
    - max `img_num` of existing data
        - plus `img_num` of transferred data
"""

import os
import glob
import shutil
import sys

def renumber_file(file_path, max_previous):
    """
    given a file path, renumber the img_num portion of the filename
    to avoid naming collision, when moving training data over to the
    test directory

    Parameters:
    ------------
    file_path: string
        file path to rename
    max_previous: int
        maximum img_num of the existing test data

    Returns:
    --------
    string
    """
    prefix = os.sep.join(file_path.split(os.sep)[:-1])
    final_path = file_path.split(os.sep)[-1]
    split_final = final_path.split("_")
    new_img_num = int(split_final[2]) + max_previous
    split_final[2] = new_img_num
    final_joined = "_".join(split_final)
    return os.path.join(prefix, final_joined)


def renumber_all_files(list_of_files, max_previous):
    """docstring"""
    new_names = [renumber_file(i, max_previous) for i in list_of_files]
    for orig, new_name in zip(list_of_files, new_names):
        os.rename(orig, new_name)


def get_img_num(file_path):
    """get img_num out of a file path"""
    return int(file_path.split(os.sep)[-1].split("_")[2])


def find_max_img_num(test_dir_path):
    """docstring"""
    all_files = glob.glob(test_dir_path + "/*/*.npy")
    all_img_nums = [get_img_num(i) for i in all_files]
    return max(all_img_nums)


if __name__ == "__main__":

    # 1. copy over entire directory structure to new destination
    SOURCE, DESTINATION = sys.argv[1:]
    shutil.copytree(SOURCE, DESTINATION)

    # 2. renumber training dataset
    destination_train_path = os.path.join(DESTINATION, "train")
    destination_test_path  = os.path.join(DESTINATION, "test")
    max_test_img_num = find_max_img_num(destination_test_path)
    train_file_list = glob.glob(destination_train_path + "/*/*.npy")
    renumber_all_files(train_file_list, max_test_img_num)

    # 3. move training dataset into test directory
    train_file_list = glob.glob(destination_train_path + "/*/*.npy")
    for filename in train_file_list:
        # rename used for moving files
        os.rename(filename, filename.replace("train", "test"))

