import os
from shutil import copyfile

import pandas as pd

import torch
import torchvision
from torchvision import transforms, datasets

from tiny_img import download_tinyImg200


def load_data():

    # Download data if necessary
    if os.path.isdir("./tiny-imagenet-200"):
        pass
    else:
        download_tinyImg200(".")

    # Create Test directory, mimicking the ImageFolder structure of train
    VAL_DIR = "./tiny-imagenet-200/val"
    TEST_DIR = "./tiny-imagenet-200/Test"

    val_list = pd.read_csv(VAL_DIR + "/val_annotations.txt", sep="\t", header=None)

    if not os.path.isdir(TEST_DIR):
        os.mkdir(TEST_DIR)

        for x in val_list.iterrows():
            img = x[1][0]
            folder = TEST_DIR + "/" + x[1][1]

            if not os.path.isdir(folder):
                os.mkdir(folder)

            copyfile(VAL_DIR + "/images" + "/" + img, folder + "/" + img)


class MapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return self.map(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


def prepare_data(val_size):
    transforms_train = transforms.Compose(
        [
            transforms.ColorJitter(hue=0.05, saturation=0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, expand=True),
            transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transforms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and split data
    full_set = datasets.ImageFolder(root="tiny-imagenet-200/train")
    test_set = datasets.ImageFolder(root="tiny-imagenet-200/Test")

    VAL_SIZE = int(len(full_set) * val_size)
    train_size, val_size = len(full_set) - VAL_SIZE, VAL_SIZE
    train_set, val_set = torch.utils.data.random_split(full_set, (train_size, val_size))

    # The class labels better be the same - else we have a problem
    assert (
        test_set.class_to_idx == full_set.class_to_idx
    ), "Test and train labels don't match"

    train_set = MapDataset(train_set, transforms_train)
    val_set = MapDataset(val_set, transforms_val)
    test_set = MapDataset(test_set, transforms_val)

    return train_set, val_set, test_set
