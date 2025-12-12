import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate
import pathlib
from tqdm import tqdm
import glob
import albumentations as A
import cv2

from PIL import Image
import pathlib
from utils import get_images_masks_paths
import shutil

if __name__ == "__main__":
    # How many augmented versions per original image
    images_to_generate = 2

    # Root of the eye dataset
    root_dir = "data/NN_human_mouse_eyes"

    # Original images and masks
    images_path = os.path.join(root_dir, "images")
    masks_path  = os.path.join(root_dir, "labels")

    # Where to save augmented data
    img_augmented_path = os.path.join(root_dir, "images_aug")
    msk_augmented_path = os.path.join(root_dir, "labels_aug")


    if os.path.exists(img_augmented_path):
        shutil.rmtree(img_augmented_path)

    if os.path.exists(msk_augmented_path):
        shutil.rmtree(msk_augmented_path)

    pathlib.Path(img_augmented_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(msk_augmented_path).mkdir(parents=True, exist_ok=True)


    print(root_dir)
    # Load image/mask pairs from data/eye_ds
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_images_masks_paths(root_dir=root_dir, sefics="")

    # Use the training split for augmentation
    images = train_x
    masks  = train_y

    # Infer file extensions from the first pair
    if len(images) == 0:
        raise RuntimeError("No training images found for augmentation.")

    ext1 = pathlib.Path(images[0]).suffix.lstrip(".") or "png"
    ext2 = pathlib.Path(masks[0]).suffix.lstrip(".") or "png"



    aug = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=1),
        A.Transpose(p=1),
        A.GridDistortion(p=1)
        ]
    )


    for i in tqdm(range(len(images)), desc = "generate images", ncols = 100):

        image = images[i]
        mask = masks[i]
      
        original_image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_mask = Image.open(mask)
        original_mask = np.array(original_mask)

        new_image_path = "%s/%s.%s" %(img_augmented_path, pathlib.Path(images[i]).stem, ext1)
        new_mask_path  = "%s/%s.%s" %(msk_augmented_path, pathlib.Path(masks[i]).stem, ext2)

        io.imsave(new_image_path, cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        io.imsave(new_mask_path, original_mask)

        for j in range(images_to_generate):     
            augmented = aug(image=original_image, mask=original_mask)
            transformed_image = augmented['image']
            transformed_mask = augmented['mask']

            new_image_path = "%s/%s_%s.%s" %(img_augmented_path, pathlib.Path(images[i]).stem, j+1, ext1)
            new_mask_path  = "%s/%s_%s.%s" %(msk_augmented_path, pathlib.Path(masks[i]).stem, j+1, ext2)

            io.imsave(new_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
            io.imsave(new_mask_path, transformed_mask)