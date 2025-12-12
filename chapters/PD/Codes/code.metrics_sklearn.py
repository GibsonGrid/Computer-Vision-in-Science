import numpy as np
from sklearn.metrics import (
    accuracy_score,
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
)

def segmentation_metrics_sklearn(y_true, y_pred, num_classes=None):
    """
    Compute segmentation metrics for binary or multi-class masks
    using scikit-learn built-in functions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels. Integer-valued array of shape (H, W) or (N, H, W).
    y_pred : np.ndarray
        Predicted labels. Same shape and dtype convention as y_true.
    num_classes : int, optional
        Number of classes (C). If None, inferred from the union of labels.

    Returns
    -------
    metrics : dict
        Dictionary containing:
            - 'pixel_accuracy'
            - 'iou_per_class'    (np.ndarray of shape (C,))
            - 'mean_iou'
            - 'dice_per_class'   (np.ndarray of shape (C,))
            - 'mean_dice'
            - 'precision_per_class' (np.ndarray of shape (C,))
            - 'recall_per_class'    (np.ndarray of shape (C,))
            - 'f1_per_class'        (np.ndarray of shape (C,))
            - 'macro_f1'
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if num_classes is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.arange(num_classes)

    # Pixel Accuracy
    pixel_accuracy = accuracy_score(y_true, y_pred)

    # IoU per class and mean IoU (Jaccard index)
    iou_per_class = jaccard_score(
        y_true, y_pred, labels=labels, average=None
    )
    mean_iou = jaccard_score(
        y_true, y_pred, labels=labels, average="macro"
    )

    # Dice per class and mean Dice:
    # For each class, Dice = F1 (with one-vs-rest formulation).
    dice_per_class = f1_score(
        y_true, y_pred, labels=labels, average=None
    )
    mean_dice = f1_score(
        y_true, y_pred, labels=labels, average="macro"
    )

    # Precision and Recall per class
    precision_per_class = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # F1 per class and macro F1 (same numbers as Dice above)
    f1_per_class = dice_per_class.copy()
    macro_f1 = mean_dice

    return {
        "pixel_accuracy": pixel_accuracy,
        "iou_per_class": iou_per_class,
        "mean_iou": mean_iou,
        "dice_per_class": dice_per_class,
        "mean_dice": mean_dice,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "macro_f1": macro_f1,
        "labels": labels,
    }


# ---------------------- Examples ----------------------
if __name__ == "__main__":
    # Binary example (0 = background, 1 = foreground)
    y_true_bin = np.array([[0, 1, 1],
                           [0, 1, 0],
                           [0, 0, 0]])
    y_pred_bin = np.array([[0, 1, 0],
                           [0, 1, 0],
                           [0, 0, 0]])

    metrics_bin = segmentation_metrics_sklearn(y_true_bin, y_pred_bin, num_classes=2)
    print("Binary segmentation metrics:")
    for k, v in metrics_bin.items():
        print(k, ":", v)

    # Multi-class example (3 classes: 0, 1, 2)
    y_true_mc = np.array([[0, 1, 2],
                          [1, 2, 0]])
    y_pred_mc = np.array([[0, 2, 2],
                          [1, 0, 0]])

    metrics_mc = segmentation_metrics_sklearn(y_true_mc, y_pred_mc, num_classes=3)
    print("\nMulti-class segmentation metrics:")
    for k, v in metrics_mc.items():
        print(k, ":", v)
