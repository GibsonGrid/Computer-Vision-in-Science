import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# =========================================================
# Helper: compute metrics for one class (binary mask)
# =========================================================
def metrics_for_class(gt, pred, cls, eps=1e-8):
    gt_c   = (gt == cls)
    pred_c = (pred == cls)

    tp = np.logical_and(gt_c, pred_c)
    fp = np.logical_and(~gt_c, pred_c)
    fn = np.logical_and(gt_c, ~pred_c)

    TP = tp.sum()
    FP = fp.sum()
    FN = fn.sum()

    iou    = TP / (TP + FP + FN + eps)
    dice   = 2 * TP / (2 * TP + FP + FN + eps)
    prec   = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)

    return iou, dice, prec, recall

def add_cell_grid(ax, linewidth=1, color='red', alpha=0.8):
    """Add visible grid lines between cells/pixels"""
    # Get data limits properly
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate exact pixel boundaries for a 5x5 grid
    x_ticks = np.arange(-0.5, 5, 1)  # 5 columns
    y_ticks = np.arange(-0.5, 5, 1)  # 5 rows
    
    # Set ticks at pixel boundaries
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Hide tick labels but keep grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Enable grid with specified style
    ax.grid(True, which='both', color=color, linewidth=linewidth, alpha=alpha, linestyle='-')
    
    # CRITICAL: Grid must be on TOP of the image
    ax.set_axisbelow(False)
    
    # Hide the spines (borders) but keep grid
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Reset limits to ensure full grid coverage
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)  # Note: ylim reversed for correct orientation
    
    # Force grid to be drawn
    ax.grid(b=True)
# =========================================================
# 1) BINARY EXAMPLE
# =========================================================

# 0 = background, 1 = foreground
gt_bin = np.array([
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
])

pred_bin = np.array([
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
])

# Pixel-wise confusion (for visualization)
tp = (gt_bin == 1) & (pred_bin == 1)
fp = (gt_bin == 0) & (pred_bin == 1)
fn = (gt_bin == 1) & (pred_bin == 0)
tn = (gt_bin == 0) & (pred_bin == 0)

labels_bin = np.zeros_like(gt_bin, dtype=np.uint8)
labels_bin[tp] = 1
labels_bin[fp] = 2
labels_bin[fn] = 3

# Metrics for foreground class = 1
iou_b, dice_b, prec_b, recall_b = metrics_for_class(gt_bin, pred_bin, cls=1)

cm_binary = ListedColormap(["black", "white"])              # GT / prediction
cm_conf   = ListedColormap(["black", "green", "red", "blue"])  # TN, TP, FP, FN

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].imshow(gt_bin, cmap=cm_binary, vmin=0, vmax=1)
axes[0].set_title("GT (Binary)")
# axes[0].axis("off")
add_cell_grid(axes[0])  # <--- ADD THIS LINE

axes[1].imshow(pred_bin, cmap=cm_binary, vmin=0, vmax=1)
axes[1].set_title("Prediction (Binary)")
# axes[1].axis("off")
add_cell_grid(axes[1])  # <--- ADD THIS LINE

axes[2].imshow(labels_bin, cmap=cm_conf, vmin=0, vmax=3)
axes[2].set_title("TP / FP / FN / TN")
# axes[2].axis("off")
add_cell_grid(axes[2])  # <--- ADD THIS LINE

legend_elements = [
    Patch(facecolor="black", label="TN"),
    Patch(facecolor="green", label="TP"),
    Patch(facecolor="red",   label="FP"),
    Patch(facecolor="blue",  label="FN"),
]
axes[2].legend(handles=legend_elements,
               loc="upper right",
               bbox_to_anchor=(1.35, 1.0))

# ---- Put metrics text INSIDE the figure (on the last panel) ----
metrics_text_bin = (
    f"IoU   = {iou_b:.2f}\n"
    f"Dice  = {dice_b:.2f}\n"
    f"Prec. = {prec_b:.2f}\n"
    f"Recall= {recall_b:.2f}"
)
axes[2].text(
    0.02, 0.02, metrics_text_bin,
    transform=axes[2].transAxes,
    fontsize=14,
    color="white",
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6)
)

plt.tight_layout()
plt.savefig("outputs/segmentation_regions_binary_with_metrics.png", dpi=500)
plt.close()


# =========================================================
# 2) MULTICLASS EXAMPLE
# =========================================================

# Classes: 0 = background, 1 = class A, 2 = class B
gt_mc = np.array([
    [0, 0, 1, 1, 0],
    [0, 1, 1, 2, 2],
    [0, 1, 2, 2, 0],
    [0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0],
])

pred_mc = np.array([
    [0, 1, 1, 1, 0],
    [0, 1, 2, 2, 2],
    [0, 1, 2, 0, 0],
    [0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0],
])

# Correct / incorrect map: 1 = correct, 0 = incorrect
correct = (gt_mc == pred_mc).astype(np.uint8)

# Colormap for 3 classes (0,1,2)
cm_classes = ListedColormap(["black", "orange", "cyan"])
cm_corr    = ListedColormap(["red", "green"])  # incorrect, correct

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].imshow(gt_mc, cmap=cm_classes, vmin=0, vmax=2)
axes[0].set_title("GT (Multiclass)")
# axes[0].axis("off")
add_cell_grid(axes[0])  # <--- ADD THIS LINE

axes[1].imshow(pred_mc, cmap=cm_classes, vmin=0, vmax=2)
axes[1].set_title("Prediction (Multiclass)")
# axes[1].axis("off")
add_cell_grid(axes[1])  # <--- ADD THIS LINE

axes[2].imshow(correct, cmap=cm_corr, vmin=0, vmax=1)
axes[2].set_title("Correct / Incorrect")
# axes[2].axis("off")
add_cell_grid(axes[2])  # <--- ADD THIS LINE

legend_mc = [
    Patch(facecolor="black",  label="Class 0 (bg)"),
    Patch(facecolor="orange", label="Class 1"),
    Patch(facecolor="cyan",   label="Class 2"),
]
axes[0].legend(handles=legend_mc,
               loc="upper right",
               bbox_to_anchor=(1.35, 1.0))

legend_corr = [
    Patch(facecolor="green", label="Correct"),
    Patch(facecolor="red",   label="Incorrect"),
]
axes[2].legend(handles=legend_corr,
               loc="upper right",
               bbox_to_anchor=(1.35, 1.0))
plt.grid()
# ----- Per-class IoU etc. -----
classes = [0, 1, 2]
iou_list    = []
dice_list   = []
prec_list   = []
recall_list = []

for c in classes:
    iou_c, dice_c, prec_c, recall_c = metrics_for_class(gt_mc, pred_mc, cls=c)
    iou_list.append(iou_c)
    dice_list.append(dice_c)
    prec_list.append(prec_c)
    recall_list.append(recall_c)

# ---- Put per-class IoU (and optionally others) on the right panel ----
metrics_lines_mc = []
for c, iou_c, dice_c in zip(classes, iou_list, dice_list):
    metrics_lines_mc.append(f"Class {c}: IoU={iou_c:.2f}, Dice={dice_c:.2f}")
metrics_text_mc = "\n".join(metrics_lines_mc)

axes[2].text(
    0.02, 0.02, metrics_text_mc,
    transform=axes[2].transAxes,
    fontsize=14,
    color="white",
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6)
)

plt.tight_layout()


plt.savefig("outputs/segmentation_regions_multiclass_with_metrics.png", dpi=500)

plt.close()

print("Saved:")
print("  segmentation_regions_binary_with_metrics.png")
print("  segmentation_regions_multiclass_with_metrics.png")



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from matplotlib.patches import Patch

# # =========================================================
# # Helper: compute metrics for one class (binary mask)
# # =========================================================
# def metrics_for_class(gt, pred, cls, eps=1e-8):
#     gt_c   = (gt == cls)
#     pred_c = (pred == cls)

#     tp = np.logical_and(gt_c, pred_c)
#     fp = np.logical_and(~gt_c, pred_c)
#     fn = np.logical_and(gt_c, ~pred_c)

#     TP = tp.sum()
#     FP = fp.sum()
#     FN = fn.sum()

#     iou    = TP / (TP + FP + FN + eps)
#     dice   = 2 * TP / (2 * TP + FP + FN + eps)
#     prec   = TP / (TP + FP + eps)
#     recall = TP / (TP + FN + eps)

#     return iou, dice, prec, recall


# # =========================================================
# # 1) BINARY EXAMPLE
# # =========================================================

# # 0 = background, 1 = foreground
# gt_bin = np.array([
#     [0, 0, 1, 1, 0],
#     [0, 1, 1, 1, 0],
#     [0, 1, 1, 0, 0],
#     [0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0],
# ])

# pred_bin = np.array([
#     [0, 1, 1, 1, 0],
#     [0, 1, 1, 0, 0],
#     [0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0],
# ])

# tp = (gt_bin == 1) & (pred_bin == 1)
# fp = (gt_bin == 0) & (pred_bin == 1)
# fn = (gt_bin == 1) & (pred_bin == 0)
# tn = (gt_bin == 0) & (pred_bin == 0)

# labels_bin = np.zeros_like(gt_bin, dtype=np.uint8)
# labels_bin[tp] = 1
# labels_bin[fp] = 2
# labels_bin[fn] = 3

# cm_binary = ListedColormap(["black", "white"])              # GT / prediction
# cm_conf   = ListedColormap(["black", "green", "red", "blue"])  # TN, TP, FP, FN

# fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# axes[0].imshow(gt_bin, cmap=cm_binary, vmin=0, vmax=1)
# axes[0].set_title("GT (Binary)")
# axes[0].axis("off")

# axes[1].imshow(pred_bin, cmap=cm_binary, vmin=0, vmax=1)
# axes[1].set_title("Prediction (Binary)")
# axes[1].axis("off")

# axes[2].imshow(labels_bin, cmap=cm_conf, vmin=0, vmax=3)
# axes[2].set_title("TP / FP / FN / TN")
# axes[2].axis("off")

# legend_elements = [
#     Patch(facecolor="black", label="TN"),
#     Patch(facecolor="green", label="TP"),
#     Patch(facecolor="red",   label="FP"),
#     Patch(facecolor="blue",  label="FN"),
# ]
# axes[2].legend(handles=legend_elements,
#                loc="upper right",
#                bbox_to_anchor=(1.35, 1.0))

# plt.tight_layout()
# plt.savefig("segmentation_regions_binary.png", dpi=300)
# plt.close()

# # Metrics for foreground class = 1
# iou, dice, prec, recall = metrics_for_class(gt_bin, pred_bin, cls=1)

# metrics_names  = ["IoU", "Dice", "Precision", "Recall"]
# metrics_values = [iou, dice, prec, recall]

# plt.figure(figsize=(5, 3))
# bars = plt.bar(metrics_names, metrics_values)
# plt.ylim(0, 1.0)
# plt.ylabel("Score")
# plt.title("Segmentation Metrics (Binary)")

# for b, v in zip(bars, metrics_values):
#     plt.text(b.get_x() + b.get_width() / 2,
#              v + 0.02,
#              f"{v:.2f}",
#              ha="center", va="bottom", fontsize=8)

# plt.tight_layout()
# plt.savefig("segmentation_metrics_binary.png", dpi=300)
# plt.close()


# # =========================================================
# # 2) MULTICLASS EXAMPLE
# # =========================================================

# # Classes: 0 = background, 1 = class A, 2 = class B
# gt_mc = np.array([
#     [0, 0, 1, 1, 0],
#     [0, 1, 1, 2, 2],
#     [0, 1, 2, 2, 0],
#     [0, 0, 2, 0, 0],
#     [0, 0, 0, 0, 0],
# ])

# pred_mc = np.array([
#     [0, 1, 1, 1, 0],
#     [0, 1, 2, 2, 2],
#     [0, 1, 2, 0, 0],
#     [0, 0, 2, 0, 0],
#     [0, 0, 0, 0, 0],
# ])

# # Correct / incorrect map: 1 = correct, 0 = incorrect
# correct = (gt_mc == pred_mc).astype(np.uint8)

# # Colormap for 3 classes (0,1,2)
# cm_classes = ListedColormap(["black", "orange", "cyan"])
# cm_corr    = ListedColormap(["red", "green"])  # incorrect, correct

# fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# axes[0].imshow(gt_mc, cmap=cm_classes, vmin=0, vmax=2)
# axes[0].set_title("GT (Multiclass)")
# axes[0].axis("off")

# axes[1].imshow(pred_mc, cmap=cm_classes, vmin=0, vmax=2)
# axes[1].set_title("Prediction (Multiclass)")
# axes[1].axis("off")

# axes[2].imshow(correct, cmap=cm_corr, vmin=0, vmax=1)
# axes[2].set_title("Correct / Incorrect")
# axes[2].axis("off")

# legend_mc = [
#     Patch(facecolor="black",  label="Class 0 (bg)"),
#     Patch(facecolor="orange", label="Class 1"),
#     Patch(facecolor="cyan",   label="Class 2"),
# ]
# axes[0].legend(handles=legend_mc,
#                loc="upper right",
#                bbox_to_anchor=(1.35, 1.0))

# legend_corr = [
#     Patch(facecolor="green", label="Correct"),
#     Patch(facecolor="red",   label="Incorrect"),
# ]
# axes[2].legend(handles=legend_corr,
#                loc="upper right",
#                bbox_to_anchor=(1.35, 1.0))

# plt.tight_layout()
# plt.savefig("segmentation_regions_multiclass.png", dpi=300)
# plt.close()

# # ----- Per-class metrics -----
# classes = [0, 1, 2]
# iou_list    = []
# dice_list   = []
# prec_list   = []
# recall_list = []

# for c in classes:
#     iou_c, dice_c, prec_c, recall_c = metrics_for_class(gt_mc, pred_mc, cls=c)
#     iou_list.append(iou_c)
#     dice_list.append(dice_c)
#     prec_list.append(prec_c)
#     recall_list.append(recall_c)

# # Example: IoU per class
# plt.figure(figsize=(5, 3))
# bars = plt.bar([str(c) for c in classes], iou_list)
# plt.ylim(0, 1.0)
# plt.xlabel("Class")
# plt.ylabel("IoU")
# plt.title("IoU per Class (Multiclass)")

# for b, v in zip(bars, iou_list):
#     plt.text(b.get_x() + b.get_width() / 2,
#              v + 0.02,
#              f"{v:.2f}",
#              ha="center", va="bottom", fontsize=8)

# plt.tight_layout()
# plt.savefig("segmentation_metrics_multiclass_iou.png", dpi=300)
# plt.close()

# print("Saved:")
# print("  segmentation_regions_binary.png")
# print("  segmentation_metrics_binary.png")
# print("  segmentation_regions_multiclass.png")
# print("  segmentation_metrics_multiclass_iou.png")


# # import numpy as np
# # import matplotlib.pyplot as plt

# # # ---------------------------------------------------------
# # # 1. Toy example: binary ground truth and prediction
# # # ---------------------------------------------------------
# # # 0 = background, 1 = foreground
# # gt = np.array([
# #     [0, 0, 1, 1, 0],
# #     [0, 1, 1, 1, 0],
# #     [0, 1, 1, 0, 0],
# #     [0, 0, 1, 0, 0],
# #     [0, 0, 0, 0, 0],
# # ])

# # pred = np.array([
# #     [0, 1, 1, 1, 0],
# #     [0, 1, 1, 0, 0],
# #     [0, 1, 0, 0, 0],
# #     [0, 0, 1, 0, 0],
# #     [0, 0, 0, 0, 0],
# # ])

# # # ---------------------------------------------------------
# # # 2. Compute TP, FP, FN, TN maps
# # # ---------------------------------------------------------
# # tp = (gt == 1) & (pred == 1)
# # fp = (gt == 0) & (pred == 1)
# # fn = (gt == 1) & (pred == 0)
# # tn = (gt == 0) & (pred == 0)

# # # Label map for visualization
# # # 0 = TN, 1 = TP, 2 = FP, 3 = FN
# # labels = np.zeros_like(gt, dtype=np.uint8)
# # labels[tp] = 1
# # labels[fp] = 2
# # labels[fn] = 3

# # # ---------------------------------------------------------
# # # 3. Visualize GT, prediction, and TP/FP/FN/TN
# # # ---------------------------------------------------------
# # from matplotlib.colors import ListedColormap

# # # Colormaps
# # cm_binary = ListedColormap(["black", "white"])          # for gt/pred
# # cm_conf   = ListedColormap(["black", "green", "red", "blue"])  # TN, TP, FP, FN

# # fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# # axes[0].imshow(gt, cmap=cm_binary, vmin=0, vmax=1)
# # axes[0].set_title("Ground Truth")
# # axes[0].axis("off")

# # axes[1].imshow(pred, cmap=cm_binary, vmin=0, vmax=1)
# # axes[1].set_title("Prediction")
# # axes[1].axis("off")

# # im = axes[2].imshow(labels, cmap=cm_conf, vmin=0, vmax=3)
# # axes[2].set_title("TP / FP / FN / TN")
# # axes[2].axis("off")

# # # Legend for TP/FP/FN/TN
# # from matplotlib.patches import Patch
# # legend_elements = [
# #     Patch(facecolor="black", label="TN"),
# #     Patch(facecolor="green", label="TP"),
# #     Patch(facecolor="red",   label="FP"),
# #     Patch(facecolor="blue",  label="FN"),
# # ]
# # axes[2].legend(handles=legend_elements,
# #                loc="upper right",
# #                bbox_to_anchor=(1.35, 1.0))

# # plt.tight_layout()
# # plt.savefig("segmentation_regions.png", dpi=300)
# # plt.close()
# # print("Saved: segmentation_regions.png")

# # # ---------------------------------------------------------
# # # 4. Compute metrics and plot bar chart
# # # ---------------------------------------------------------
# # TP = tp.sum()
# # FP = fp.sum()
# # FN = fn.sum()
# # TN = tn.sum()

# # epsilon = 1e-8
# # iou     = TP / (TP + FP + FN + epsilon)
# # dice    = 2 * TP / (2 * TP + FP + FN + epsilon)
# # prec    = TP / (TP + FP + epsilon)
# # recall  = TP / (TP + FN + epsilon)

# # metrics_names  = ["IoU", "Dice", "Precision", "Recall"]
# # metrics_values = [iou, dice, prec, recall]

# # plt.figure(figsize=(5, 3))
# # bars = plt.bar(metrics_names, metrics_values)
# # plt.ylim(0, 1.0)
# # plt.ylabel("Score")
# # plt.title("Segmentation Metrics (Binary Example)")

# # # Add value on top of each bar
# # for b, v in zip(bars, metrics_values):
# #     plt.text(b.get_x() + b.get_width() / 2,
# #              v + 0.02,
# #              f"{v:.2f}",
# #              ha="center", va="bottom", fontsize=8)

# # plt.tight_layout()
# # plt.savefig("segmentation_metrics_barplot.png", dpi=300)
# # plt.close()
# # print("Saved: segmentation_metrics_barplot.png")
