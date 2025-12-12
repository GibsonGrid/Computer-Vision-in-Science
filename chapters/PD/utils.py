
import os
import time
import random
import numpy as np
import cv2
import torch
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

import glob
""" Seeding the randomness. """

def get_CLANE(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img[:,:,0] = clahe.apply(img[:,:,0])
    img[:,:,1] = clahe.apply(img[:,:,1])
    img[:,:,2] = clahe.apply(img[:,:,2])
    return img


    
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




# #---------------------------MODIFY
# def get_images_masks_paths(dataset_name=None, sefics="_aug", root_dir="data/eye_ds",
#                            test_size=0.2, val_size=0.2):
#     """
#     Read all image/mask pairs from:
#         root_dir/images
#         root_dir/labels

#     Images and masks are matched by filename stem (before extension).
#     The dataset is randomly split into train/valid/test.
#     """

#     root_dir = pathlib.Path(root_dir)
#     images_path = root_dir / f"images{sefics}"
#     masks_path = root_dir / f"labels{sefics}"

#     assert images_path.is_dir(), f"Images folder not found: {images_path}"
#     assert masks_path.is_dir(), f"Labels folder not found: {masks_path}"

#     # Collect all image files
#     image_files = sorted(list(images_path.rglob("*.*")))
#     df = pd.DataFrame(columns=["image", "mask"])

#     for img_fn in image_files:
#         # find a label with the same stem in labels/
#         candidates = list(masks_path.rglob(f"{img_fn.stem}.*"))
#         # print(str(img_fn), str(img_fn)[0:2] != "H-")
#         if len(candidates) == 0 or str(img_fn)[0:2] != "H-":
#             # no matching mask for this image; skip
#             continue

#         df.loc[len(df)] = [str(img_fn), str(candidates[0])]

#     if len(df) == 0:
#         raise RuntimeError(f"No matched image/mask pairs found under {root_dir}")

#     # Split into train / val / test
#     # First: train+val vs test
#     train_val_df, test_df = train_test_split(
#         df, test_size=test_size, random_state=42, shuffle=True
#     )
#     # Then split train_val into train and val
#     val_rel = val_size / (1.0 - test_size)  # relative val size inside (train+val)
#     train_df, val_df = train_test_split(
#         train_val_df, test_size=val_rel, random_state=42, shuffle=True
#     )

#     train_x = list(train_df["image"])
#     train_y = list(train_df["mask"])
#     valid_x = list(val_df["image"])
#     valid_y = list(val_df["mask"])
#     test_x  = list(test_df["image"])
#     test_y  = list(test_df["mask"])

#     return train_x, train_y, valid_x, valid_y, test_x, test_y
def get_images_masks_paths(sefics,
                           root_dir="data/eye_ds",
                           test_size=0.2, val_size=0.2):
    """
    Read all image/mask pairs from:
        root_dir/images
        root_dir/labels

    - Only keep images whose filename starts with 'H-'
    - Only keep pairs where BOTH image and mask exist.
    - Images and masks are matched by filename stem.
    - Dataset is split into train/valid/test.

    Returns:
        train_x, train_y, valid_x, valid_y, test_x, test_y
    """

    import pathlib
    import pandas as pd
    from sklearn.model_selection import train_test_split

    root_dir     = pathlib.Path(root_dir)
    images_path  = root_dir / f"images{sefics}"
    masks_path   = root_dir / f"labels{sefics}"

    assert images_path.is_dir(), f"Images folder not found: {images_path}"
    assert masks_path.is_dir(), f"Labels folder not found: {masks_path}"

    image_files = sorted(images_path.rglob("H-*.jpg"))

    mask_files = sorted(masks_path.rglob("H-*.png"))
    
    print(f"Found: {len(image_files)} images, {len(mask_files)} masks")

    df = pd.DataFrame(columns=["image", "mask"])

    df["image"] = [str(e) for e in image_files]
    df["mask"] = [str(e) for e in image_files]


    # ---------------- DATA SPLIT ----------------
    # train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, shuffle=True
    )

    # relative validation size inside train_val
    val_rel = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df, test_size=val_rel, random_state=42, shuffle=True
    )

    return (
        list(train_df["image"]), list(train_df["mask"]),
        list(val_df["image"]),   list(val_df["mask"]),
        list(test_df["image"]),  list(test_df["mask"]),
    )



def append2csv(filename, df):
    df.to_csv(filename, mode='a', index=False, header = not pathlib.Path(filename).exists())




def build_gif(image_folder, image_files = None, where2save = None):
    if image_files is None:
        image_files = list(pathlib.Path(image_folder).rglob(f"*.*png"))[0:10]

    # Create a list to hold the images
    images = []
    for filename in image_files:
        #filepath = os.path.join(image_folder, filename)
        images.append(imageio.imread(filename))

    # Save as gif
    imageio.mimsave(f'{where2save}/output.gif', images, duration=0.7)  # duration is the time between frames in second