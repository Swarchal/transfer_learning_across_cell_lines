"""
module docstring
"""
import os
import sys
import torch
import glob
from random import shuffle
import numpy as np
from PIL import Image
from skimage import io
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt


def load_model(state_dict_path):
    """load a pretrained model"""
    model_state_dict = torch.load(state_dict_path)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model.cuda()


scale = transforms.Scale(256)
crop = transforms.CenterCrop(224)
to_tensor = transforms.ToTensor()
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_image(img_path):
    """load a single image and label"""
    image = Image.open(img_path)
    transformed_image = scale(image)
    transformed_image = crop(transformed_image)
    transformed_image = to_tensor(transformed_image)
    transformed_image = norm(transformed_image)
    # different array dimensions to training as we don't have batches anymore
    # so this dimension can be removed from the tensor
    transformed_image.unsqueeze_(0)
    label = img_path.split(os.sep)[-2]
    return [transformed_image, label]


def find_classes(dir_name):
    """return a dictionary relating class number to the class name"""
    classes = os.listdir(dir_name)
    classes.sort()
    class_to_idx = {i: classes[i] for i in range(len(classes))}
    return class_to_idx


def predict(model, image, label_dict):
    """
    Given a model, single image and the label dictionary:
    Return the predicted class of the image.
    """
    image = Variable(image).cuda()
    ans = model(image)
    ans = ans.data.cpu().numpy()[0]
    ans = softmax(ans)
    idx_max = np.argmax(ans)
    predicted = label_dict[idx_max]
    return predicted


def softmax(x):
     """Compute softmax values for each sets of scores in x."""
     e_x = np.exp(x - np.max(x))
     return e_x / e_x.sum()


def get_img_number(img_path):
    """record image number on predictions,
    useful for the consenus classification"""
    final_path = img_path.split(os.sep)[-1]
    return int(final_path.split("_")[1])


if __name__ == "__main__":

    cell_line = sys.argv[1]
    if cell_line not in ["MDA-231", "MDA-157", "MCF7", "HCC1954", "HCC1569", "SKBR3", "T47D", "KPL4"]:
        raise ValueError("{} is not a valid cell-line".format(cell_line))

    # use train dir, not used in model training and more images than test dir
    # TODO: combine train and test dir to increase the datasize
    IMG_DIR = "/exports/eddie/scratch/s1027820/chopped/nncell_data_300_{}/test".format(cell_line)
    MODEL_PATH = "/exports/igmm/eddie/Drug-Discovery/scott/pytorch_stuff/models/2018-03-29_nn_loo_models/{}_excluded_trained_model_adam.pynn".format(cell_line)
    img_list = glob.glob(IMG_DIR + "/*/*")
    shuffle(img_list)
    label_dict = find_classes(IMG_DIR)
    model = load_model(MODEL_PATH)
    output_path = "/exports/igmm/eddie/Drug-Discovery/scott/pytorch_stuff/transfer_learning/predictions/2018-03-29_repeat/{}_predictions.csv"
    with open(output_path.format(cell_line), "w") as f:
        f.write("actual,predicted,img_num\n")
        for idx, image_path in enumerate(img_list):
                image, label = load_image(image_path)
                img_num = get_img_number(image_path)
                predicted = predict(model, image, label_dict)
                f.write("{},{},{}\n".format(label, predicted, img_num))

