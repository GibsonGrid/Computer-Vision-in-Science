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
# 2) MULTICLASS EXAMPLE WITH CONFUSION MATRIX TABLE
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

# Create figure with 2 rows: 1 for images, 1 for table
fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.5], hspace=0.05)

# Create a sub-grid for the top row with 3 columns
top_gs = gs[0].subgridspec(1, 3, wspace=0.1)
bottom_ax = fig.add_subplot(gs[1])

# Create axes for images
top_axes = []
for i in range(3):
    ax = fig.add_subplot(top_gs[0, i])
    top_axes.append(ax)

# Plot images
top_axes[0].imshow(gt_mc, cmap=cm_classes, vmin=0, vmax=2)
top_axes[0].set_title("GT (Multiclass)", fontsize=12, pad=10)
add_cell_grid(top_axes[0])

top_axes[1].imshow(pred_mc, cmap=cm_classes, vmin=0, vmax=2)
top_axes[1].set_title("Prediction (Multiclass)", fontsize=12, pad=10)
add_cell_grid(top_axes[1])

top_axes[2].imshow(correct, cmap=cm_corr, vmin=0, vmax=1)
top_axes[2].set_title("Correct / Incorrect", fontsize=12, pad=10)
add_cell_grid(top_axes[2])

# Add legends
legend_mc = [
    Patch(facecolor="black",  label="Class 0 (bg)"),
    Patch(facecolor="orange", label="Class 1"),
    Patch(facecolor="cyan",   label="Class 2"),
]
top_axes[0].legend(handles=legend_mc,
                   # loc="lower right",
                   ncol =1,
                   bbox_to_anchor=(0.65, -0.1),
                   fontsize=9)

legend_corr = [
    Patch(facecolor="green", label="Correct"),
    Patch(facecolor="red",   label="Incorrect"),
]
top_axes[2].legend(handles=legend_corr,
                   # loc="upper right",
                   bbox_to_anchor=(1, -0.15),
                   fontsize=9)

# ----- Compute confusion matrix for each class -----
def compute_confusion_matrix(gt, pred, cls):
    """Compute TP, FP, FN, TN for a specific class"""
    gt_c = (gt == cls)
    pred_c = (pred == cls)
    
    tp = np.logical_and(gt_c, pred_c).sum()
    fp = np.logical_and(~gt_c, pred_c).sum()
    fn = np.logical_and(gt_c, ~pred_c).sum()
    tn = np.logical_and(~gt_c, ~pred_c).sum()
    
    return tp, fp, fn, tn

# Compute for all classes
classes = [0, 1, 2]
confusion_data = []
for c in classes:
    tp, fp, fn, tn = compute_confusion_matrix(gt_mc, pred_mc, c)
    confusion_data.append([tp, fp, fn, tn])

# Also compute metrics
iou_list = []
dice_list = []
for c in classes:
    iou_c, dice_c, _, _ = metrics_for_class(gt_mc, pred_mc, cls=c)
    iou_list.append(iou_c)
    dice_list.append(dice_c)

# ----- Create table in bottom row -----
bottom_ax.axis('tight')
bottom_ax.axis('off')

# Table data
table_data = []
headers = ['Class', 'TP', 'FP', 'FN', 'TN', 'IoU', 'Dice']

for i, c in enumerate(classes):
    tp, fp, fn, tn = confusion_data[i]
    table_data.append([
        f'Class {c}',
        f'{tp}',
        f'{fp}',
        f'{fn}',
        f'{tn}',
        f'{iou_list[i]:.2f}',
        f'{dice_list[i]:.2f}'
    ])

# Create table
table = bottom_ax.table(
    cellText=table_data,
    colLabels=headers,
    cellLoc='center',
    loc='center',
    colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)  # Scale cell size

# Color header row
for j in range(len(headers)):
    table[(0, j)].set_facecolor('#404040')
    table[(0, j)].set_text_props(weight='bold', color='white')

# Color alternating rows for readability
for i in range(len(classes)):
    color = '#f0f0f0' if i % 2 == 0 else '#ffffff'
    for j in range(len(headers)):
        table[(i+1, j)].set_facecolor(color)

# Add title to table
bottom_ax.set_title('Per-Class Confusion Matrix and Metrics', fontsize=13, pad=0.5, weight='bold')

# ----- Put per-class IoU on the right panel (optional) -----
metrics_lines_mc = []
for c, iou_c, dice_c in zip(classes, iou_list, dice_list):
    metrics_lines_mc.append(f"Class {c}: IoU={iou_c:.2f}, Dice={dice_c:.2f}")
metrics_text_mc = "\n".join(metrics_lines_mc)

top_axes[2].text(
    0.02, 0.02, metrics_text_mc,
    transform=top_axes[2].transAxes,
    fontsize=10,
    color="white",
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6)
)

plt.tight_layout()
plt.savefig("outputs/segmentation_regions_multiclass_with_table.png", dpi=500, bbox_inches='tight')
plt.close()

print("Saved: segmentation_regions_multiclass_with_table.png")