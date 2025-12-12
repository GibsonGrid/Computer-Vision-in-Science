from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import pathlib
# from utils import create_dir




# Build dictionaries: base name -> path
def build_index(paths, suffix_to_strip=""):
    idx = {}
    for p in paths:
        stem = p.stem
        if suffix_to_strip and stem.endswith(suffix_to_strip):
            stem = stem[: -len(suffix_to_strip)]
        idx[stem] = p
    return idx



def buildResults(resDir):
        # ---------------------------------------------------------------------
        # Collect paths
        # ---------------------------------------------------------------------
        images_org_paths = list(pathlib.Path(f"{resDir}/images_org").rglob("*.*"))
        images_clane_paths = list(pathlib.Path(f"{resDir}/images_CLANE").rglob("*.*"))
        masks_paths = list(pathlib.Path(f"{resDir}/masks").rglob("*.*"))
        preds_paths = list(pathlib.Path(f"{resDir}/prediction").rglob("*.*"))
        ellipse_overlay_paths = list(pathlib.Path(f"{resDir}/ellipse_overlay").rglob("*.*"))



        idx_org = build_index(images_org_paths)
        idx_clane = build_index(images_clane_paths)
        idx_masks = build_index(masks_paths)
        idx_preds = build_index(preds_paths)
        # ellipse overlay files are saved as "<name>_overlay.png"
        idx_ellipse = build_index(ellipse_overlay_paths, suffix_to_strip="_overlay")

        # Intersection of names present in ALL folders
        common_names = (
            set(idx_org.keys())
            & set(idx_clane.keys())
            & set(idx_masks.keys())
            & set(idx_preds.keys())
            & set(idx_ellipse.keys())
        )

        common_names = sorted(list(common_names))
        if len(common_names) == 0:
            raise RuntimeError("No common filenames across folders. Check outputs.")

        # Shuffle order (random permutation)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(common_names))
        names_ordered = [common_names[i] for i in perm]

        # ---------------------------------------------------------------------
        # Build lists of images in the NEW consistent order
        # ---------------------------------------------------------------------
        images = []
        images_clane = []
        masks = []
        preds = []
        images_clane_masks_preds = []
        ellipse_overlay_list = []

        for name in names_ordered:
            img_fn = idx_org[name]
            img_clane_fn = idx_clane[name]
            mask_fn = idx_masks[name]
            pred_fn = idx_preds[name]
            ellipse_fn = idx_ellipse[name]

            img_org = cv2.imread(str(img_fn))
            img_clane = cv2.imread(str(img_clane_fn))
            mask = cv2.imread(str(mask_fn))
            pred_mask = cv2.imread(str(pred_fn))
            ellipse_img = cv2.imread(str(ellipse_fn))

            if img_org is None or img_clane is None or mask is None or pred_mask is None:
                # Skip inconsistent or unreadable entries
                continue
            if ellipse_img is None:
                # Skip if ellipse overlay not readable (should not happen, but safe)
                continue

            images.append(img_org)
            images_clane.append(img_clane)
            masks.append(mask)
            preds.append(pred_mask)
            ellipse_overlay_list.append(ellipse_img)

            # build overlay: CLANE + GT + XOR with prediction (same as before)
            pred_mask_tmp = pred_mask.copy()
            pred_mask_tmp[:, :, 0] = 0
            pred_mask_tmp[:, :, 2] = pred_mask_tmp[:, :, 1]

            mask_tmp = mask.copy()
            mask_tmp[:, :, 0] = 0
            mask_tmp[:, :, 2] = 0

            tmp = 0.6 * img_clane + 0.8 * mask_tmp
            tmp = 0.6 * tmp + 0.9 * np.bitwise_xor(mask_tmp, pred_mask)
            images_clane_masks_preds.append(tmp)

        # ---------------------------------------------------------------------
        # Now all lists have the same length = number of valid common samples
        # ---------------------------------------------------------------------
        N = len(images)
        if N == 0:
            raise RuntimeError("No valid samples after filtering common images.")

        print(f"Number of common samples: {N}")

        # ---------------------------------------------------------------------
        # ds_samples1.png  (original, CLANE, mask)
        # ---------------------------------------------------------------------
        n = min(5, N)
        res1 = np.concatenate(images[0:n], axis=1)
        res2 = np.concatenate(images_clane[0:n], axis=1)
        res3 = np.concatenate(masks[0:n], axis=1)
        res = np.concatenate([res1, res2, res3], axis=0)
        cv2.imwrite(f"{resDir}/ds_samples1.png", res)

        res = np.concatenate([res1, res2], axis=0)
        cv2.imwrite(f"{resDir}/ds_samples_CLANE.png", res)

        # ---------------------------------------------------------------------
        # model1_results_output_3efore*.png
        #  - three rows of 5 images each (if available)
        # ---------------------------------------------------------------------
        rows = 3
        cols = 5
        needed = rows * cols
        available = min(needed, len(images_clane_masks_preds), len(ellipse_overlay_list))

        # If fewer than needed, adjust rows accordingly
        cols = min(cols, available)
        rows = max(1, available // cols)

        # First build CLANE+mask+pred mosaic
        blocks1 = images_clane_masks_preds[: rows * cols]
        rows_img1 = [
            np.concatenate(blocks1[r * cols:(r + 1) * cols], axis=1) for r in range(rows)
        ]
        res = np.concatenate(rows_img1, axis=0)
        cv2.imwrite(f"{resDir}/model1_results_output_3efore.png", res)

        # Then ellipse-overlay mosaic
        blocks2 = ellipse_overlay_list[: rows * cols]
        rows_img2 = [
            np.concatenate(blocks2[r * cols:(r + 1) * cols], axis=1) for r in range(rows)
        ]
        res = np.concatenate(rows_img2, axis=0)
        cv2.imwrite(f"{resDir}/model1_results_output_3efore____.png", res)

        print("Done creating mosaics.")



if __name__ == "__main__":
        resDir = r"Z:\Road detection_mohammed_exps\out_dataset1"
        buildResults(resDir)

