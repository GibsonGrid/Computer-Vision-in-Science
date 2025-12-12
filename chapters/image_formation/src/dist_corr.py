import glob
import cv2

def draw_grid(image, color=(0, 255, 0), thickness=1):
    """
    Draws a grid overlay on the given image.

    Args:
        image (numpy.ndarray): The input image on which the grid will be drawn.
        color (tuple, optional): The color of the grid lines in BGR format. Defaults to (0, 255, 0) (green).
        thickness (int, optional): The thickness of the grid lines. Defaults to 1.

    Returns:
        numpy.ndarray: The image with the grid overlay.
    """
    h, w = image.shape[:2]

    grid_size_w = w / 20  
    grid_size_h = h / 10  
    
    grid_size_w = max(1, int(round(grid_size_w)))
    grid_size_h = max(1, int(round(grid_size_h)))
    
    # Draw vertical lines
    for x in range(0, w + 1, grid_size_w):  
        cv2.line(image, (x, 0), (x, h), color, thickness)

    # Draw horizontal lines
    for y in range(0, h + 1, grid_size_h):  
        cv2.line(image, (0, y), (w, y), color, thickness)

    return image

save_folder = "../calib_imgs"
cv_file = cv2.FileStorage(f"{save_folder}/calib_cam.xml", cv2.FILE_STORAGE_READ)
cameraMatrix = cv_file.getNode("K").mat()
dist = cv_file.getNode("dist").mat()
images = glob.glob(f'{save_folder}/*.png')
img_id = 0
img = cv2.imread(images[img_id])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

img = draw_grid(img, color=(0, 255, 0), thickness=1)
cv2.imwrite("../outputs/orig.png", img)

undst = cv2.undistort(img, cameraMatrix, dist, None, newcameramtx)
cv2.imwrite("../outputs/undist.png", undst)