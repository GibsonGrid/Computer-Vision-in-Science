import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A

def main():
    # --------- User settings --------- #
    image_path = "test_data/eye_example.png"   # path to your input image
    save_dir   = "augmented_examples_25"       # folder to save augmented images
    os.makedirs(save_dir, exist_ok=True)

    # --------- Read image --------- #
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --------- Define 25 augmentations --------- #
    augmentations = [
        ("horizontal_flip", A.HorizontalFlip(p=1.0)),
        ("vertical_flip", A.VerticalFlip(p=1.0)),
        ("random_rotate_90", A.RandomRotate90(p=1.0)),
        ("shift_scale_rotate", A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                                 rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1.0)),
        ("grid_distortion", A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)),
        ("elastic_transform", A.ElasticTransform(alpha=50, sigma=8, alpha_affine=10, p=1.0)),
        ("optical_distortion", A.OpticalDistortion(distort_limit=0.3, shift_limit=0.05, p=1.0)),
        # ("random_brightness_contrast", A.RandomBrightnessContrast(brightness_limit=0.2,
        #                                                           contrast_limit=0.2, p=1.0)),
        ("hue_saturation_value", A.HueSaturationValue(hue_shift_limit=10,
                                                      sat_shift_limit=15,
                                                      val_shift_limit=10, p=1.0)),
        ("rgb_shift", A.RGBShift(r_shift_limit=15, g_shift_limit=40, b_shift_limit=15, p=1.0)),
        ("channel_shuffle", A.ChannelShuffle(p=1.0)),
        ("clahe", A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)),
        ("blur", A.Blur(blur_limit=5, p=1.0)),
        ("motion_blur", A.MotionBlur(blur_limit=7, p=1.0)),
        ("median_blur", A.MedianBlur(blur_limit=15, p=1.0)),
        ("sharpen", A.Sharpen(alpha=(0.2, 0.5), lightness=(0.8, 1.2), p=1.0)),
        ("emboss", A.Emboss(alpha=(0.2, 0.5), strength=(0.4, 0.7), p=1.0)),
        ("random_gamma", A.RandomGamma(gamma_limit=(80, 200), p=1.0)),
        ("gauss_noise", A.GaussNoise(var_limit=(20.0, 50.0), p=1.0)),
        ("multiplicative_noise", A.MultiplicativeNoise(multiplier=(0.9, 1.5), per_channel=True, p=1.0)),
        ("downscale", A.Downscale(scale_min=0.5, scale_max=0.8, interpolation=cv2.INTER_NEAREST, p=1.0)),
        ("perspective", A.Perspective(scale=(0.02, 0.05), keep_size=True, p=1.0)),
        ("coarse_dropout", A.CoarseDropout(max_holes=8, max_height=20, max_width=20,
                                           min_holes=1, fill_value=0, p=1.0)),
        ("equalize", A.Equalize(mode="cv", by_channels=True, p=1.0)),
        ("color_jitter", A.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.5, hue=0.2, p=1.0)),
    ]

    aug_images_rgb = []
    aug_names = []

    # --------- Apply each augmentation once --------- #
    for name, aug in augmentations:
        augmented = aug(image=img_bgr)  
        img_aug_bgr = augmented["image"]
        img_aug_rgb = cv2.cvtColor(img_aug_bgr, cv2.COLOR_BGR2RGB)

        aug_images_rgb.append(img_aug_rgb)
        aug_names.append(name)

        out_path = os.path.join(save_dir, f"{name}.png")
        cv2.imwrite(out_path, img_aug_bgr)

    # Save original
    orig_save_path = os.path.join(save_dir, "original.png")
    cv2.imwrite(orig_save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # ============================================================
    #  NEW PART: CREATE ONE BIG GRID IMAGE (original + 25 aug)
    # ============================================================
    print("Creating collage (grid) image...")

    all_images = [img_rgb] + aug_images_rgb
    titles = ["original"] + aug_names

    cell_h, cell_w = img_rgb.shape[:2]
    n_total = len(all_images)
    n_cols = 5
    n_rows = int(np.ceil(n_total / n_cols))

    # Create blank big canvas
    grid_h = n_rows * cell_h
    grid_w = n_cols * cell_w
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255  # white background

    # Paste each augmented image into the big collage
    for idx, img in enumerate(all_images):
        r = idx // n_cols
        c = idx % n_cols
        y1, y2 = r * cell_h, (r + 1) * cell_h
        x1, x2 = c * cell_w, (c + 1) * cell_w
        grid[y1:y2, x1:x2] = img

    # Save the grid image
    collage_path = os.path.join(save_dir, "all_augmented_grid.png")
    cv2.imwrite(collage_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"GRID SAVED: {collage_path}")

    # Show the collage
    plt.figure(figsize=(20, 15))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()

    print(f"Saved original and {len(aug_images_rgb)} augmented images to: {save_dir}")

if __name__ == "__main__":
    main()
