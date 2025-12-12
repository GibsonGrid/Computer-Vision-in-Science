import numpy as np
import cv2
import glob

save_folder = "../calib_imgs"

# Define pattern size you generated and camera resoution
# Pattern size is taking crosses in each axes, not the exact number of squares
pattern_size = (10, 7)
frameSize = (1920, 1080)

# Termination criteria for calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for pattern, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = [] # 3d points of the pattern
imgpoints = [] # 2d points

# Pathes of captured frames
images = glob.glob(f'{save_folder}/*.png')

for img in images:

    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret == True:
        objpoints.append(objp)

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners with index of each corner
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        for i, pos in enumerate(corners.reshape(-1, 2)):
            cv2.putText(img, str(i), pos.astype(int)+5, font,
                        fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("Img", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Calibration results
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, frameSize, None, None)
print("K", cameraMatrix)
print("Distortion coefficients", dist)

cv_file = cv2.FileStorage(f"{save_folder}/calib_cam.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("K", cameraMatrix)
cv_file.write("dist", dist)
cv_file.release()
