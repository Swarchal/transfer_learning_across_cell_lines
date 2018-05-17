"""
author: Scott Warchal
date: 2018-04-24
"""

import os
import glob
import numpy as np
from skimage import transform as ski_transform
import torch
import torch.utils.data
from torch import Tensor


class CellDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for structured directory of numpy arrays to work
    with torch.utils.DataLoader

    Directory structure should mirror that of ImageFolder. i.e:

        all_data
        ├── test
        │   ├── actin
        │   ├── aurora
        │   ├── dna_damaging
        │   ├── kinase
        │   ├── microtubule
        │   ├── protein_deg
        │   ├── protein_synth
        │   └── statin
        └── train
            ├── actin
            ├── aurora
            ├── dna_damaging
            ├── kinase
            ├── microtubule
            ├── protein_deg
            ├── protein_synth
            └── statin

    So you would have a CellDataset for train and test separately.
    Storing these in a dicionary would be the sensible thing to do. e.g:

        path = "/path/to/all_data"
        datasets = {x: CellDataset(os.path.join(path, x) for x in ["train", "test])}
    """
    def __init__(self, data_dir, transforms=None, return_name=False):
        self.data_dir = data_dir
        self.image_list = self.get_image_list(data_dir)
        self.label_dict = self.generate_label_dict(self.image_list)
        self.labels = self.get_unique_labels(self.image_list)
        self.transforms = transforms
        self.return_name = return_name

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = np.load(self.image_list[index])
        img = self.reshape(img)
        if self.transforms is not None:
            img = self.transforms(img)
        label_name = self.get_class_label(self.image_list[index])
        label_index = torch.IntTensor([self.label_dict[label_name]])
        if self.return_name:
            name = self.image_list[index].split(os.sep)[-1]
            return img, label_index, name
        else:
            return img, label_index

    @staticmethod
    def get_image_list(data_dir):
        """generate list of numpy arrays from the data directory"""
        all_images = glob.glob(os.path.join(data_dir, "*/*"))
        return [i for i in all_images if i.endswith(".npy")]

    def get_unique_labels(self, img_list):
        """
        return a list that just contains strings of all the class labels
        in a sorted order
        """
        all_labels = [self.get_class_label(i) for i in img_list]
        return sorted(list(set(all_labels)))

    def generate_label_dict(self, image_list):
        """create dictionary of {class_name: index}"""
        all_labels = [self.get_class_label(i) for i in image_list]
        unique_sorted_labels = sorted(list(set(all_labels)))
        return {label: int(i) for i, label in enumerate(unique_sorted_labels)}

    @staticmethod
    def get_class_label(img_path):
        """get MOA label from the file path
        return this as an integer index"""
        img_path = os.path.abspath(img_path)
        return img_path.split(os.sep)[-2]

    @staticmethod
    def reshape(img):
        """
        reshape a 300x300x5 numpy array into
        a 1x5x244x244 torch Tensor, as ResNet expects 244*244 tensors.
        """
        # resize image to from 300*300*5 => 244*244*5, also converts to float
        img = ski_transform.resize(img, (244, 244, 5), mode="reflect")
        # reshape from width*height*channel => channel*width*height
        return Tensor(img).permute(2, 0, 1)


def make_datasets(top_level_data_dir, transforms=None, **kwargs):
    """
    Parameters:
    -----------
    top_level_data_dir: string
        directory path which contains train and test sub-directories
    transforms: transformation dictionary (default is None)
        if not None, then should be a dictionary of transformations, with
        an entry for training transforms and testing transforms.

    Returns:
    --------
    Dictionary containing
        {"train": CellDataset,
         "test" : CellDataset}
    """
    dataset_dict = {}
    for phase in ["train", "test"]:
        dataset_path = os.path.join(top_level_data_dir, phase)
        if transforms is not None:
            print("INFO: images will be randomly rotated in training")
            dataset_dict[phase] = CellDataset(
                dataset_path,
                transforms=transforms if phase == "train" else None,
                **kwargs
            )
        else:
            dataset_dict[phase] = CellDataset(dataset_path, **kwargs)
    return dataset_dict


def make_dataloaders(datasets_dict, batch_size=32, num_workers=8):
    """
    Parameters:
    -----------
    datasets_dict: dictionary
        dictionary created from make_datasets()
    batch_size: int
        number of images per batch
    num_workers: int
        number of sub-processes used to pre-load the images

    Returns:
    --------
    Dictionary of DataLoaders
        {"train": DataLoader,
         "test" : DataLoader}
    """
    dataloader_dict = {}
    for phase in ["train", "test"]:
        dataloader_dict[phase] = torch.utils.data.DataLoader(
            datasets_dict[phase],
            batch_size=batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=num_workers
        )
    return dataloader_dict
