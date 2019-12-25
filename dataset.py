import random
import os
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, Resize, Normalize, RandomCrop


class UnpairedImageDataset(Dataset):
    def __init__(self, domain_X, domain_Y, image_size, root="./data"):
        self.transform = Compose([  RandomCrop(image_size),
                                    RandomHorizontalFlip(p=0.5),
                                    ToTensor(),
                                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.files_X = sorted(glob(os.path.join(root, domain_X) + '/*.*'))
        self.files_Y = sorted(glob(os.path.join(root, domain_Y) + '/*.*'))

    def __getitem__(self, index):
        item_X = Image.open(self.files_X[index])
        item_Y = Image.open(self.files_Y[index])

        if self.transform is not None:
            item_X = self.transform(item_X)
            item_Y = self.transform(item_Y)

        return item_X, item_Y

    def __len__(self):
        return min(len(self.files_X), len(self.files_Y))
    
    def shuffle(self):
        random.shuffle(self.files_X)
        random.shuffle(self.files_Y)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0)
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return, dim=0)