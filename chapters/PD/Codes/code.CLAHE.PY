
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_CLAHE(img_rgb):
    """
    Apply CLANE (here: CLAHE-based contrast-limited adaptive normalization)
    channel-wise to an RGB image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = img_rgb.copy()
    out[:, :, 0] = clahe.apply(img_rgb[:, :, 0])
    out[:, :, 1] = clahe.apply(img_rgb[:, :, 1])
    out[:, :, 2] = clahe.apply(img_rgb[:, :, 2])
    return out

if __name__ == "__main__":
    # -------------------- Load image --------------------
    image_path = "test_data/eye_example.png"  # set your path here
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    create_dir("outputs")

    # BGR -> RGB for plotting
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # -------------------- Apply CLANE --------------------
    img_clane_rgb = get_CLAHE(img_rgb)

    # -------------------- Show/save images --------------------
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("After CLANE")
    plt.imshow(img_clane_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/clane_images.pdf", dpi=300)
    plt.close()

    # -------------------- Histograms (grayscale) --------------------
    # Convert to grayscale for intensity histograms
    gray_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_clane = cv2.cvtColor(img_clane_rgb, cv2.COLOR_RGB2GRAY)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Histogram (Original)")
    plt.hist(gray_orig.ravel(), bins=256, range=(0, 255))
    plt.xlabel("Intensity")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.title("Histogram (After CLANE)")
    plt.hist(gray_clane.ravel(), bins=256, range=(0, 255))
    plt.xlabel("Intensity")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("outputs/clane_histograms.pdf", dpi=300)
    plt.close()
