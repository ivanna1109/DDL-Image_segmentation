import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pickle
from u_net_model import UNet
from sklearn.model_selection import train_test_split

class ImageMaskDataset():
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask

# Load data from pickle file
def read_data():
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

        images_dict = data['images']
        masks_dict = data['masks']

# Pretvaranje podataka u liste numpy nizova
        images = [images_dict[key] for key in images_dict]
        masks = [masks_dict[key] for key in masks_dict]

        images = np.array(images)
        masks = np.array(masks)
        return images, masks

