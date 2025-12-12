
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import pathlib
from PIL import Image
from utils import get_CLANE

class customDataset(Dataset):
    def __init__(self, images_path, masks_path, size, mask2gray = 1):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.size = size
        self.mask2gray = mask2gray

    def __getitem__(self, index):
        """ Reading image """
        # print(os.path.isfile(self.images_path[index]),
        #     os.path.isfile(self.masks_path[index])
        #     )
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = get_CLANE(cv2.resize(image, self.size))
        image = image/255.0 ## (512, 512, 3)
        

        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        # print(image.shape)
        """ Reading mask """
        # print(self.masks_path[index])
        # print("------------------------------- ", index, len(self.masks_path), self.masks_path[index])
        # print("Image: ", self.images_path[index], image.shape)
        # print("*" * 50)

        # print("Mask: ", self.masks_path[index])
        
        # mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        # mask = cv2.resize(mask, self.size)
        

        # mask = Image.open(self.masks_path[index])
        # # print(np.array(mask).shape)
        # # if self.mask2gray:
        # mask = mask.resize(self.size).convert('L')
        # mask = np.array(mask)
        # # print(mask.shape)

        # mask = mask/255.0   ## (512, 512)
        # mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        # # print(mask.shape)
        # mask = mask.astype(np.float32)
        # mask = torch.from_numpy(mask)

        mask = Image.open(self.masks_path[index])
        mask = mask.resize(self.size).convert('L')
        mask = np.array(mask)

        # Binarize: any non-zero value is treated as foreground
        mask = (mask > 0).astype(np.float32)  # values in {0.0, 1.0}

        mask = np.expand_dims(mask, axis=0)   # (1, H, W)
        mask = torch.from_numpy(mask)


        
        
        # print("Mask: ", self.masks_path[index], mask.shape)
        return image, mask

    def __len__(self):
        return self.n_samples
