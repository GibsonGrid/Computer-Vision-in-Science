import cv2
import numpy as np
import os


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_pupil_geometry(mask: np.ndarray,
                           do_morphology: bool = True):
    """
    Extract pupil center and ellipse axes from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary pupil mask (2D). Pupil = 1 or 255, background = 0.
    do_morphology : bool
        Whether to apply a small morphological closing to clean the mask.

    Returns
    -------
    x : float
        X-coordinate of the ellipse center (column index).
    y : float
        Y-coordinate of the ellipse center (row index).
    dminor : float
        Length of minor axis (in pixels).
    dmajor : float
        Length of major axis (in pixels).
    """
    # Ensure uint8 binary mask {0, 255}
    mask = np.asarray(mask)
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # Optional: small closing to remove holes / noise
    if do_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find external contours
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise RuntimeError("No contour found in mask (is the pupil empty?).")

    # Take the largest contour as the pupil
    cnt = max(contours, key=cv2.contourArea)

    # fitEllipse requires at least 5 points
    if len(cnt) < 5:
        raise RuntimeError("Not enough points to fit an ellipse.")

    # Fit ellipse: center (x, y), axes (major, minor), angle
    (x, y), (axis1, axis2), angle = cv2.fitEllipse(cnt)

    # Identify major / minor axis length
    dmajor = max(axis1, axis2)  # major axis length (diameter)
    dminor = min(axis1, axis2)  # minor axis length (diameter)

    return x, y, dminor, dmajor


# ============================================================
# Example usage on a mask image file
# ============================================================
if __name__ == "__main__":
    out_dir = "outputs/"
    create_dir(out_dir)

    # Load mask (white pupil on black, or any non-zero pupil)
    mask_path = "test_data/pupil_mask.png"
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    x, y, dminor, dmajor = extract_pupil_geometry(mask_img)

    print(f"Pupil center: x={x:.2f}, y={y:.2f}")
    print(f"Minor axis:  dminor={dminor:.2f} pixels")
    print(f"Major axis:  dmajor={dmajor:.2f} pixels")

    # Optional: visualize fitted ellipse over the mask
    vis = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    center = (int(round(x)), int(round(y)))
    axes = (int(round(dmajor / 2)), int(round(dminor / 2)))
    angle = 0  # we don’t use it here, but fitEllipse returned it above

    # Re-fit to get angle for drawing:
    contours, _ = cv2.findContours((mask_img > 0).astype(np.uint8)*255,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), (a1, a2), ang = cv2.fitEllipse(cnt)
    center = (int(round(cx)), int(round(cy)))
    axes = (int(round(a1 / 2)), int(round(a2 / 2)))
    angle = ang

    cv2.ellipse(vis, center, axes, angle, 0, 360, (0, 0, 255), 2)
    cv2.circle(vis, center, 2, (0, 255, 0), -1)

    cv2.imwrite(f"{out_dir}/pupil_with_ellipse.png", vis)
