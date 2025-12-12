import numpy as np

def segmentation_metrics(y_true, y_pred, num_classes=None, eps=1e-8):
    """
    Compute segmentation metrics for binary or multi-class masks.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels. Integer-valued array of shape (H, W) or (N, H, W).
    y_pred : np.ndarray
        Predicted labels. Same shape and dtype convention as y_true.
    num_classes : int, optional
        Number of classes (C). If None, inferred from max label in y_true and y_pred.
    eps : float, optional
        Small value to avoid division by zero.

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
    # Flatten
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    # Confusion matrix n_ij:
    # rows = true class i, columns = predicted class j
    cm = np.bincount(
        num_classes * y_true + y_pred,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

    # Pixel Accuracy (multi-class definition)
    pixel_accuracy = np.trace(cm) / (cm.sum() + eps)

    # Per-class statistics
    TP = np.diag(cm).astype(float)
    FP = cm.sum(axis=0) - TP   # predicted as class i, but not true i
    FN = cm.sum(axis=1) - TP   # true class i, but predicted not i
    TN = cm.sum() - (TP + FP + FN)

    # IoU per class
    iou_per_class = TP / (TP + FP + FN + eps)
    mean_iou = np.mean(iou_per_class)

    # Dice per class
    dice_per_class = 2 * TP / (2 * TP + FP + FN + eps)
    mean_dice = np.mean(dice_per_class)

    # Precision, Recall, F1 per class
    precision_per_class = TP / (TP + FP + eps)
    recall_per_class    = TP / (TP + FN + eps)
    f1_per_class = 2 * precision_per_class * recall_per_class / (
        precision_per_class + recall_per_class + eps
    )
    macro_f1 = np.mean(f1_per_class)

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

    metrics_bin = segmentation_metrics(y_true_bin, y_pred_bin, num_classes=2)
    print("Binary segmentation metrics:")
    for k, v in metrics_bin.items():
        print(k, ":", v)

    # Multi-class example (3 classes: 0, 1, 2)
    y_true_mc = np.array([[0, 1, 2],
                          [1, 2, 0]])
    y_pred_mc = np.array([[0, 2, 2],
                          [1, 0, 0]])

    metrics_mc = segmentation_metrics(y_true_mc, y_pred_mc, num_classes=3)
    print("\nMulti-class segmentation metrics:")
    for k, v in metrics_mc.items():
        print(k, ":", v)
