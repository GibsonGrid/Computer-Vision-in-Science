import os
import time
from operator import add
import argparse
import pathlib

import cv2
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from model import get_model
from utils import (
    create_dir,
    seeding,
    get_images_masks_paths,
    get_CLANE,
)


# ============================================================
# Metric utilities
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Compute basic metrics on (N,1,H,W) tensors."""
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_true = (y_true > 0.5).astype(np.uint8).reshape(-1)
    y_pred = (y_pred > 0.5).astype(np.uint8).reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-7)

    return [
        score_jaccard,
        score_f1,
        score_recall,
        score_precision,
        score_acc,
        specificity,
    ]


def calculate_metrics_as_dict(y_true, y_pred):
    """Return metrics as dict (for averaging)."""
    METRICS = {
        "iou": 0,
        "f1_score": 0,
        "precision": 0,
        "sensitivity": 0,
        "specificity": 0,
        "accuracy": 0,
    }

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_true = (y_true > 0.5).astype(np.uint8).reshape(-1)
    y_pred = (y_pred > 0.5).astype(np.uint8).reshape(-1)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-7)

    METRICS["iou"] = jaccard_score(y_true, y_pred, labels=[0, 1])
    METRICS["f1_score"] = f1_score(y_true, y_pred, labels=[0, 1])
    METRICS["sensitivity"] = recall_score(y_true, y_pred, labels=[0, 1])
    METRICS["precision"] = precision_score(y_true, y_pred, labels=[0, 1])
    METRICS["accuracy"] = accuracy_score(y_true, y_pred)
    METRICS["specificity"] = specificity

    return METRICS


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


# ============================================================
# Pupil ellipse post-processing (shared by both modes)
# ============================================================

def extract_pupil_geometry(mask, do_morphology=True):
    """
    Extract pupil center and ellipse axes from a predicted binary mask.

    Parameters
    ----------
    mask : np.ndarray
        2D mask, pupil > 0, background = 0.
    do_morphology : bool
        Apply small morphological closing to clean the mask.

    Returns
    -------
    x, y : float
        Ellipse center (image coordinates).
    dminor, dmajor : float
        Minor- and major-axis lengths (diameters in pixels).
    (cnt, (cx, cy), (axis1, axis2), angle) : tuple
        Extra info for visualization.
    """
    mask = np.asarray(mask)
    mask = (mask > 0).astype(np.uint8) * 255

    if do_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        raise RuntimeError("No contour found in predicted mask.")

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        raise RuntimeError("Not enough points to fit an ellipse.")

    (x, y), (axis1, axis2), angle = cv2.fitEllipse(cnt)

    dmajor = max(axis1, axis2)
    dminor = min(axis1, axis2)

    return x, y, dminor, dmajor, (cnt, (x, y), (axis1, axis2), angle)


# ============================================================
# Argument parser
# ============================================================

def create_parser():
    parser = argparse.ArgumentParser(
        description="Segmentation testing script (dataset or single image)."
    )

    parser.add_argument("--where2save", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--W", type=int, required=True)
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--DATASET_NAME", type=str, default="eye_ds")
    parser.add_argument("--MODEL_ARCH", type=str, required=True)
    parser.add_argument("--ENCODER_NAME", type=str, required=True)
    parser.add_argument("--N_SAMPLES", type=int, default=-1)
    parser.add_argument("--USE_AUGMENTED", type=int)


    parser.add_argument(
        "--mode",
        type=str,
        choices=["dataset", "single"],
        default="dataset",
        help="dataset: full test set; single: one image + pupil features",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to a single image (required if --mode single).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold on sigmoid output for binarization.",
    )

    return parser


# ============================================================
# DATASET EVALUATION
# ============================================================

def eval_dataset(
    test_x,
    test_y,
    checkpoint_path,
    model_info,
    model,
    where2save,
    size,
    threshold=0.5,
    test_yd=None,
    has_gt=True,          # <--- NEW FLAG
):
    """
    If has_gt = True:
        - expects test_y to contain mask paths
        - computes segmentation metrics
        - saves GT masks & comparison panels
    If has_gt = False:
        - ignores test_y
        - runs pure inference (prediction + ellipse features only)
        - returns simple summary dict
    """
    seeding(42)
    create_dir(where2save)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_ARCH, ENCODER_NAME = model_info

    if model is None:
        model = get_model(MODEL_ARCH, ENCODER_NAME, 1)
        model = model.to(device)
        if checkpoint_path:
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=device)
            )

    model.eval()

    metrics_result_new = {
        "iou": [],
        "f1_score": [],
        "precision": [],
        "sensitivity": [],
        "specificity": [],
        "accuracy": [],
    }
    metrics_score = [0.0] * 6
    time_taken = []
    y_pred_diag = []

    H, W = size

    parent = pathlib.Path(where2save).parent
    create_dir(f"{parent}/images_org")
    create_dir(f"{parent}/images_CLANE")
    create_dir(f"{parent}/prediction")
    if has_gt:
        create_dir(f"{parent}/masks")

    # Folders for ellipse overlay and features
    ellipse_overlay_dir = f"{parent}/ellipse_overlay"
    ellipse_feat_dir = f"{parent}/ellipse_features"
    create_dir(ellipse_overlay_dir)
    create_dir(ellipse_feat_dir)

    # Ensure test_y has same length if provided
    if test_y is None:
        test_y = [None] * len(test_x)

    for i, (x_path, y_path) in tqdm(
        enumerate(zip(test_x, test_y)), total=len(test_x)
    ):
        print(x_path, y_path)
        name = pathlib.Path(x_path).stem

        # Original + CLANE
        image_org = cv2.imread(x_path, cv2.IMREAD_COLOR)
        if image_org is None:
            print(f"Warning: could not read image {x_path}, skipping.")
            continue

        image_org = cv2.resize(image_org, (W, H))
        image = get_CLANE(image_org.copy())

        x = np.transpose(image, (2, 0, 1))
        x = x / 255.0
        x = np.expand_dims(x, axis=0).astype(np.float32)
        x = torch.from_numpy(x).to(device)

        with torch.no_grad():
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            pred_y_np = pred_y[0, 0].cpu().numpy()
            pred_y_bin = (pred_y_np > threshold).astype(np.uint8)

        # Save org / CLANE / prediction for all cases
        pred_vis = mask_parse(pred_y_bin)
        cv2.imwrite(f"{parent}/images_org/{name}.png", image_org)
        cv2.imwrite(f"{parent}/images_CLANE/{name}.png", image)
        cv2.imwrite(f"{parent}/prediction/{name}.png", pred_vis * 255)

        # If GT is available, compute metrics and save GT panels
        if has_gt and y_path is not None and os.path.exists(str(y_path)):
            mask = cv2.imread(str(y_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: could not read mask {y_path}, skipping metrics.")
            else:
                mask = cv2.resize(mask, (W, H))
                y_mask_np = (mask > 0).astype(np.float32)
                y_mask = torch.from_numpy(
                    y_mask_np[None, None, ...]
                ).to(device)

                score_dict = calculate_metrics_as_dict(y_mask, pred_y[0])
                for k in metrics_result_new.keys():
                    metrics_result_new[k].append(score_dict[k])

                score = calculate_metrics(y_mask, pred_y)
                metrics_score = list(map(add, metrics_score, score))

                y_pred_diag.append(int(np.max(pred_y_bin) > 0))

                ori_mask = mask_parse(mask)
                line = np.ones((H, 3, 3)) * 100
                cat_images = np.concatenate(
                    [image, line, ori_mask, line, pred_vis * 255], axis=1
                )
                cv2.imwrite(f"{where2save}/{name}.png", cat_images)
                cv2.imwrite(f"{parent}/masks/{name}.png", ori_mask)
        else:
            # Inference-only: save CLANE + prediction panel
            line = np.ones((H, 3, 3)) * 100
            cat_images = np.concatenate(
                [image, line, pred_vis * 255], axis=1
            )
            cv2.imwrite(f"{where2save}/{name}.png", cat_images)

        # Ellipse + features from predicted mask
        mask_path = os.path.join(ellipse_feat_dir, f"{name}_mask.png")
        cv2.imwrite(mask_path, pred_y_bin * 255)

        features_path = os.path.join(ellipse_feat_dir, f"{name}_features.txt")
        overlay_path = os.path.join(ellipse_overlay_dir, f"{name}_overlay.png")

        try:
            x_c, y_c, dminor, dmajor, ellipse_info = extract_pupil_geometry(
                pred_y_bin
            )
            cnt, (cx, cy), (axis1, axis2), angle = ellipse_info

            with open(features_path, "w") as f:
                f.write("# Pupil features from predicted mask\n")
                f.write(f"x_center {x_c:.4f}\n")
                f.write(f"y_center {y_c:.4f}\n")
                f.write(f"dminor {dminor:.4f}\n")
                f.write(f"dmajor {dmajor:.4f}\n")

            vis = image_org.copy()
            center = (int(round(cx)), int(round(cy)))
            axes = (int(round(axis1 / 2)), int(round(axis2 / 2)))
            cv2.ellipse(vis, center, axes, angle, 0, 360, (0, 0, 255), 2)
            cv2.circle(vis, center, 3, (0, 255, 0), -1)
            cv2.imwrite(overlay_path, vis)

        except RuntimeError as e:
            with open(features_path, "w") as f:
                f.write("# Pupil features from predicted mask\n")
                f.write("# Extraction failed: " + str(e) + "\n")

    # ------------------------------------------------------------------
    # Final results
    # ------------------------------------------------------------------
    if not has_gt:
        # Pure inference: simple summary
        res = {
            "num_images": [len(time_taken)],
            "mean_inference_time_sec": [float(np.mean(time_taken)) if len(time_taken) > 0 else 0.0],
        }
        return res

    # Original metric aggregation when GT exists
    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    spec = metrics_score[5] / len(test_x)

    if test_yd is not None:
        test_yd = list(test_yd)
        score_f1_diag = f1_score(test_yd, y_pred_diag)
        score_recall_diag = recall_score(test_yd, y_pred_diag)
        score_precision_diag = precision_score(test_yd, y_pred_diag)
        score_acc_diag = accuracy_score(test_yd, y_pred_diag)

        metrics_ = [
            jaccard,
            f1,
            recall,
            precision,
            acc,
            score_f1_diag,
            score_recall_diag,
            score_precision_diag,
            score_acc_diag,
        ]
        res = pd.DataFrame(metrics_).T
        res.columns = [
            "jaccard",
            "f1",
            "recall",
            "precision",
            "acc",
            "f1_diag",
            "recall_diag",
            "precision_diag",
            "acc_diag",
        ]
    else:
        avg_metrics = {}
        for k, v in metrics_result_new.items():
            avg_metrics[k] = [np.mean(np.array(v).flatten())]
        print(avg_metrics)
        res = avg_metrics

    return res



def run_dataset_mode(args):
    DATASET_NAME = args.DATASET_NAME
    where2save = args.where2save
    checkpoint_path = args.checkpoint_path
    MODEL_ARCH = args.MODEL_ARCH
    ENCODER_NAME = args.ENCODER_NAME
    N_SAMPLES = args.N_SAMPLES
    USE_AUGMENTED = args.USE_AUGMENTED
    H, W = args.H, args.W

    size = (H, W)

    create_dir(where2save)

    # ------------------------------------------------------------------
    # Branch: MMU-Iris-Database => inference on images only
    # ------------------------------------------------------------------
    ds_name = (DATASET_NAME or "").lower()

    if ds_name.startswith("mmu"):
        # Root folder for MMU-Iris-Database
        mmu_root = pathlib.Path("Data/MMU-Iris-Database")

        if not mmu_root.is_dir():
            raise FileNotFoundError(f"MMU root folder not found: {mmu_root}")

        IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        all_imgs = []

        # Expected structure:
        # MMU-Iris-Database/
        #   subject_1/
        #       left/
        #           *.png / *.jpg / ...
        #       right/
        #           ...
        #   subject_2/
        #       ...
        for subj_dir in sorted(mmu_root.iterdir()):
            if not subj_dir.is_dir():
                continue

            for side_name in ["left", "right", "Left", "Right", "L", "R"]:
                side_dir = subj_dir / side_name
                if not side_dir.is_dir():
                    continue
                for ext in IMG_EXTS:
                    all_imgs.extend(sorted(side_dir.glob(f"*{ext}")))

        if len(all_imgs) == 0:
            raise RuntimeError(f"No images found under {mmu_root}")

        if N_SAMPLES > 0:
            all_imgs = all_imgs[:N_SAMPLES]

        test_x = [str(p) for p in all_imgs]
        test_y = [None] * len(test_x)

        print(f"Found {len(test_x)} images from MMU-Iris-Database.")
        print((MODEL_ARCH, ENCODER_NAME))

        res = eval_dataset(
            test_x,
            test_y,
            checkpoint_path,
            (MODEL_ARCH, ENCODER_NAME),
            None,
            where2save + "/samples",
            size,
            threshold=args.threshold,
            test_yd=None,
            has_gt=False,   # <--- inference only
        )

        # Simple summary CSV (num images + mean inference time)
        pd.DataFrame(res).to_csv(
            f"{where2save}/mmu_inference_summary.csv", index=False
        )

        print("MMU-Iris-Database inference finished.")
        return

    # ------------------------------------------------------------------
    # Default branch: original dataset with masks (evaluation mode)
    # ------------------------------------------------------------------
    USE_AUGMENTED = 0
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_images_masks_paths(root_dir=f"data/{DATASET_NAME}", sefics="_aug" if USE_AUGMENTED else "")


    if N_SAMPLES > 0:
        test_x = test_x[:N_SAMPLES]
        test_y = test_y[:N_SAMPLES]

    print((MODEL_ARCH, ENCODER_NAME))
    res = eval_dataset(
        test_x,
        test_y,
        checkpoint_path,
        (MODEL_ARCH, ENCODER_NAME),
        None,
        where2save + "/samples",
        size,
        threshold=args.threshold,
        test_yd=None,
        has_gt=True,   # <--- original behaviour with GT
    )

    pd.DataFrame(res).to_csv(f"{where2save}/metrics_test.csv")

    resDir = where2save
    create_dir(resDir)

    # Optional: keep the final_results.csv handling
    if os.path.exists("final_results.csv"):
        df = pd.read_csv("final_results.csv")
        df = df[
            [
                "MODEL_ARCH",
                "ENCODER_NAME",
                "iou",
                "f1_score",
                "precision",
                "sensitivity",
                "specificity",
                "accuracy",
                "model_size",
                "num_epochs",
                "lr",
                "epoch_for_best",
                "Exp",
            ]
        ]
        df = df.sort_values(
            ["iou", "f1_score", "specificity"],
            ascending=[False, False, False],
        )
        df.to_csv(f"{resDir}/final_res_updated.csv", index=False)

    images_org_paths = list(pathlib.Path(f"{where2save}/images_org").rglob("*.*"))
    images_clane_paths = list(pathlib.Path(f"{where2save}/images_CLANE").rglob("*.*"))
    masks_paths = list(pathlib.Path(f"{where2save}/masks").rglob("*.*"))
    preds_paths = list(pathlib.Path(f"{where2save}/prediction").rglob("*.*"))

    images = []
    images_clane = []
    masks = []
    preds = []
    images_clane_masks_preds = []

    for i in np.random.randint(0, len(images_org_paths), len(images_org_paths)):
        img_fn = images_org_paths[i]
        img_clane_fn = images_clane_paths[i]
        mask_fn = masks_paths[i]
        pred_fn = preds_paths[i]

        img_org = cv2.imread(str(img_fn))
        img_clane = cv2.imread(str(img_clane_fn))
        mask = cv2.imread(str(mask_fn))
        pred_mask = cv2.imread(str(pred_fn))

        images.append(img_org)
        images_clane.append(img_clane)
        masks.append(mask)
        preds.append(pred_mask)

        pred_mask_tmp = pred_mask.copy()
        pred_mask_tmp[:, :, 0] = 0
        pred_mask_tmp[:, :, 2] = pred_mask_tmp[:, :, 1]

        mask_tmp = mask.copy()
        mask_tmp[:, :, 0] = 0
        mask_tmp[:, :, 2] = 0
        tmp = 0.6 * img_clane + 0.8 * mask_tmp
        tmp = 0.6 * tmp + 0.9 * np.bitwise_xor(mask_tmp, pred_mask)
        images_clane_masks_preds.append(tmp)

    # ds_samples1.png
    n = min(5, len(images))
    res1 = np.concatenate(images[0:n], axis=1)
    res2 = np.concatenate(images_clane[0:n], axis=1)
    res3 = np.concatenate(masks[0:n], axis=1)
    res = np.concatenate([res1, res2, res3], axis=0)
    cv2.imwrite(f"{resDir}/ds_samples1.png", res)

    # model1_results_output_3.png
    if len(images_clane_masks_preds) >= 10:
        n = 2
        res_tmp1 = np.concatenate(images_clane_masks_preds[0:n], axis=1)
        res_tmp2 = np.concatenate(images_clane_masks_preds[n:n + n], axis=1)
        res_tmp3 = np.concatenate(images_clane_masks_preds[n + n:2 * n + n], axis=1)
        res_tmp4 = np.concatenate(images_clane_masks_preds[2 * n + n:3 * n + n], axis=1)
        res_tmp5 = np.concatenate(images_clane_masks_preds[3 * n + n:4 * n + n], axis=1)
        res = np.concatenate(
            [res_tmp1, res_tmp2, res_tmp3, res_tmp4, res_tmp5], axis=0
        )
        cv2.imwrite(f"{resDir}/model1_results_output_3.png", res)

    # model1_results_output_4.png
    if len(images_clane_masks_preds) >= 8:
        n = 4
        res_tmp1 = np.concatenate(images_clane_masks_preds[0:n], axis=1)
        res_tmp2 = np.concatenate(images_clane_masks_preds[n:n + n], axis=1)
        res = np.concatenate([res_tmp1, res_tmp2], axis=0)
        cv2.imwrite(f"{resDir}/model1_results_output_4.png", res)

    print("Dataset mode finished.")
    from build_results import buildResults
    buildResults(resDir)

# ============================================================
# SINGLE-IMAGE MODE
# ============================================================

def run_single_mode(args):
    if args.image_path is None:
        raise ValueError("In single mode, --image_path must be provided.")

    where2save = args.where2save
    checkpoint_path = args.checkpoint_path
    MODEL_ARCH = args.MODEL_ARCH
    ENCODER_NAME = args.ENCODER_NAME
    H, W = args.H, args.W
    thr = args.threshold

    create_dir(where2save)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(MODEL_ARCH, ENCODER_NAME, 1)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    img_bgr = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    img_name = pathlib.Path(args.image_path).stem
    img_org = img_bgr.copy()

    img_bgr = cv2.resize(img_bgr, (W, H))
    img_clane = get_CLANE(img_bgr.copy())

    x = img_clane.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x).to(device)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
        pred = pred[0, 0].cpu().numpy()

    pred_bin = (pred > thr).astype(np.uint8)
    mask_bin = (pred_bin * 255).astype(np.uint8)

    mask_path = os.path.join(where2save, f"{img_name}_mask.png")
    cv2.imwrite(mask_path, mask_bin)

    features_path = os.path.join(where2save, f"{img_name}_features.txt")
    overlay_path = os.path.join(where2save, f"{img_name}_overlay.png")

    try:
        x_c, y_c, dminor, dmajor, ellipse_info = extract_pupil_geometry(pred_bin)
        cnt, (cx, cy), (axis1, axis2), angle = ellipse_info

        with open(features_path, "w") as f:
            f.write("# Pupil features from predicted mask\n")
            f.write(f"x_center {x_c:.4f}\n")
            f.write(f"y_center {y_c:.4f}\n")
            f.write(f"dminor {dminor:.4f}\n")
            f.write(f"dmajor {dmajor:.4f}\n")

        vis = cv2.resize(img_org, (W, H))
        center = (int(round(cx)), int(round(cy)))
        axes = (int(round(axis1 / 2)), int(round(axis2 / 2)))
        cv2.ellipse(vis, center, axes, angle, 0, 360, (0, 0, 255), 2)
        cv2.circle(vis, center, 3, (0, 255, 0), -1)
        cv2.imwrite(overlay_path, vis)

        print("Pupil features:")
        print(f"  x_center = {x_c:.2f}")
        print(f"  y_center = {y_c:.2f}")
        print(f"  dminor   = {dminor:.2f}")
        print(f"  dmajor   = {dmajor:.2f}")
        print(f"Mask saved to      : {mask_path}")
        print(f"Features saved to  : {features_path}")
        print(f"Overlay saved to   : {overlay_path}")

    except RuntimeError as e:
        with open(features_path, "w") as f:
            f.write("# Pupil features from predicted mask\n")
            f.write("# Extraction failed: " + str(e) + "\n")
        print("Warning:", e)
        print(f"Binary mask saved to {mask_path}, but no valid ellipse was found.")
        print(f"Details written to  {features_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if args.mode == "dataset":
        run_dataset_mode(args)
    else:
        run_single_mode(args)
