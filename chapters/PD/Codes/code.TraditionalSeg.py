"""
Classical image segmentation methods applied to the same image.

This script demonstrates:
- Threshold-based segmentation:
    * Global threshold
    * Otsu's threshold
    * Adaptive/local threshold
- Edge-based segmentation:
    * Sobel gradient magnitude
    * Canny edges
- Region-based segmentation:
    * Simple region growing (flood fill)
    * Watershed segmentation
- Clustering-based segmentation:
    * K-means clustering on color image

It:
- Shows all results in a Matplotlib figure.
- Saves each result as a separate image.
- Saves one concatenated image that contains all methods
  stacked vertically, with a title above each one.

Requirements:
    pip install opencv-python matplotlib scikit-learn
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



# =========================
# Utility functions
# =========================

def load_image(path):
    """Load image from path in BGR format and convert to RGB and grayscale."""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image from '{path}'")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_rgb, img_gray


# =========================
# Thresholding methods
# =========================

def global_threshold(gray, thresh=127):
    """Global (manual) threshold."""
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


def otsu_threshold(gray):
    """Otsu's automatic threshold."""
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def adaptive_threshold(gray, block_size=11, C=2):
    """Adaptive (local) threshold."""
    # Ensure block_size is odd and >=3
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )
    return binary


# =========================
# Edge-based methods
# =========================

def sobel_edges(gray):
    """Sobel gradient magnitude."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
    return mag


def canny_edges(gray, low_thresh=100, high_thresh=200):
    """Canny edge detector."""
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return edges


# =========================
# Region-based methods
# =========================

def region_growing_floodfill(gray, seed=None, lo_diff=5, up_diff=5):
    """
    Simple region growing using OpenCV floodFill.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (uint8).
    seed : tuple or None
        Seed point (x, y). If None, use center of image.
    lo_diff, up_diff : int
        Intensity differences allowed for region growing.
    """
    h, w = gray.shape
    if seed is None:
        seed = (w // 2, h // 2)  # (x, y)

    # Flood fill requires a mask 2 pixels larger than the image
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = gray.copy()

    # Fill with white (255) starting from seed
    cv2.floodFill(filled, mask, seedPoint=seed,
                  newVal=255,
                  loDiff=(lo_diff,),
                  upDiff=(up_diff,))
    # Create a binary mask of the filled region
    region = (filled == 255).astype(np.uint8) * 255
    return region


def watershed_segmentation(img_rgb, gray):
    """
    Watershed segmentation: suitable for separating touching/overlapping objects.

    Steps:
    - Otsu threshold
    - Morphological opening
    - Distance transform
    - Marker labeling
    - Watershed
    """
    # Binary image via Otsu
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological opening to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distance transform for sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform,
        0.5 * dist_transform.max(),
        255, 0
    )
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    num_markers, markers = cv2.connectedComponents(sure_fg)
    # Ensure markers start from 1, background from 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed expects BGR image
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.watershed(img_bgr, markers)

    # Boundaries are marked with -1
    seg = img_rgb.copy()
    seg[markers == -1] = [255, 0, 0]  # mark boundaries in red

    return seg


# =========================
# Clustering-based methods
# =========================

def kmeans_segmentation(img_rgb, k=3, max_iter=100):
    """
    K-means clustering segmentation on color image.

    Pixels are grouped into k clusters in RGB space.
    """
    h, w, c = img_rgb.shape
    X = img_rgb.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=max_iter, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    segmented = centers[labels].reshape(h, w, 3)
    return segmented


# =========================
# Demo
# =========================

if __name__ == "__main__":
    # Path to input image
    IMAGE_PATH = "test_data/eye_example.png"  # change this to your image path

    # Load image
    img_rgb, img_gray = load_image(IMAGE_PATH)

    # --- Threshold-based segmentation ---
    thr_global = global_threshold(img_gray, thresh=127)
    thr_otsu = otsu_threshold(img_gray)
    thr_adapt = adaptive_threshold(img_gray, block_size=21, C=5)

    # --- Edge-based segmentation ---
    sobel_mag = sobel_edges(img_gray)
    canny = canny_edges(img_gray, low_thresh=100, high_thresh=200)

    # --- Region-based segmentation ---
    region_grow = region_growing_floodfill(img_gray)
    watershed_seg = watershed_segmentation(img_rgb, img_gray)

    # --- Clustering-based segmentation ---
    kmeans_seg = kmeans_segmentation(img_rgb, k=3)

    # =========================
    # Visualization using Matplotlib (grid view)
    # =========================
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.ravel()

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (RGB)", fontsize = 16)
    axes[0].axis("off")

    axes[1].imshow(thr_global, cmap="gray")
    axes[1].set_title("Global threshold", fontsize = 16)
    axes[1].axis("off")

    axes[2].imshow(thr_otsu, cmap="gray")
    axes[2].set_title("Otsu threshold", fontsize = 16)
    axes[2].axis("off")

    axes[3].imshow(thr_adapt, cmap="gray")
    axes[3].set_title("Adaptive threshold", fontsize = 16)
    axes[3].axis("off")

    axes[4].imshow(sobel_mag, cmap="gray")
    axes[4].set_title("Sobel gradient", fontsize = 16)
    axes[4].axis("off")

    axes[5].imshow(canny, cmap="gray")
    axes[5].set_title("Canny edges", fontsize = 16)
    axes[5].axis("off")

    axes[6].imshow(region_grow, cmap="gray")
    axes[6].set_title("Region growing (flood fill)", fontsize = 16)
    axes[6].axis("off")

    axes[7].imshow(watershed_seg)
    axes[7].set_title("Watershed segmentation", fontsize = 16)
    axes[7].axis("off")

    axes[8].imshow(kmeans_seg)
    axes[8].set_title("K-means segmentation", fontsize = 16)
    axes[8].axis("off")

    # Leave extra slots blank if fewer than 12 methods
    for i in range(9, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    # =========================
    # Save individual results to disk
    # =========================
    out_dir = "outputs"
    create_dir(out_dir)


    # Save using OpenCV (convert RGB->BGR where needed)
    # cv2.imwrite(os.path.join(out_dir, "original.png"),
    #             cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(os.path.join(out_dir, "thr_global.png"), thr_global)
    # cv2.imwrite(os.path.join(out_dir, "thr_otsu.png"), thr_otsu)
    # cv2.imwrite(os.path.join(out_dir, "thr_adaptive.png"), thr_adapt)
    # cv2.imwrite(os.path.join(out_dir, "sobel_mag.png"), sobel_mag)
    # cv2.imwrite(os.path.join(out_dir, "canny.png"), canny)
    # cv2.imwrite(os.path.join(out_dir, "region_growing.png"), region_grow)
    # cv2.imwrite(os.path.join(out_dir, "watershed.png"),
    #             cv2.cvtColor(watershed_seg, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(os.path.join(out_dir, "kmeans_seg.png"),
    #             cv2.cvtColor(kmeans_seg, cv2.COLOR_RGB2BGR))

    print(f"Segmentation results saved in folder: {out_dir}")

    # =========================
    # Create and save a concatenated montage of all results
    # =========================
    # =========================
    # Create and save a 3×3 grid of all segmentation results
    # =========================

    print("Creating 3×3 segmented image grid ...")

    images_list = [
        ("Original", img_rgb),
        ("Global Threshold", thr_global),
        ("Otsu Threshold", thr_otsu),
        ("Adaptive Threshold", thr_adapt),
        ("Sobel", sobel_mag),
        ("Canny", canny),
        ("Region Growing", region_grow),
        ("Watershed", watershed_seg),
        ("K-means", kmeans_seg),
    ]

    # Create a 3×3 plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    for idx, (title, img) in enumerate(images_list):
        ax = axes[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=18)
        ax.axis("off")

    plt.tight_layout()

    # Save the 3x3 figure
    grid_path = os.path.join(out_dir, "all_methods_grid.png")
    plt.savefig(grid_path, dpi=200)
    plt.close()

    print(f"3×3 grid saved to: {grid_path}")
