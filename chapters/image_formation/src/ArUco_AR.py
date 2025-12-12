import cv2
import numpy as np
from pynput import keyboard
from project_trans import project_points_manual

save_folder = "../calib_imgs"

cv_file = cv2.FileStorage()
cv_file.open(f"{save_folder}/calib_cam.xml", cv2.FileStorage_READ)
camera_matrix = np.array(cv_file.getNode("K").mat(), dtype=np.float64)
dist_coeffs = np.array(cv_file.getNode("dist").mat(), dtype=np.float64)
cv_file.release()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

cube_size = 1

# Define plane perpendicular to marker
video_plane_width = 2
video_plane_height = 2

video_plane_points = np.array([
    [video_plane_width / 2, 0,  video_plane_height / 2],  # right-top
    [-video_plane_width / 2, 0,  video_plane_height / 2],   # left-top
    [-video_plane_width / 2, 0, -video_plane_height / 2],  # left-bottom
    [video_plane_width / 2, 0, -video_plane_height / 2],  # right-bottom
], dtype=np.float32)

save = False
terminate = False


def on_key_press(key):
    global terminate
    if key == keyboard.Key.esc:
        print("###### TERMINATE APPLICATION ######")
        terminate = True

listener = keyboard.Listener(on_press=on_key_press)
listener.start()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# In other case
# cap = cv2.VideoCapture(0)

width = 1920
height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M","J","P","G")) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for i, corner in enumerate(corners):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner, cube_size, camera_matrix, dist_coeffs)

            R, _ = cv2.Rodrigues(rvec)

            video_frame = cv2.flip(frame, 1)
            video_resized = cv2.resize(
                video_frame, (int(video_plane_width * 1000), int(video_plane_height * 1000)))

            src_pts = np.float32([[0, 0], [video_resized.shape[1], 0], [
                                    video_resized.shape[1], video_resized.shape[0]], [0, video_resized.shape[0]]])

            # Shift video plane if needed
            video_points_offset = video_plane_points.copy()
            video_points_offset[:, 2] += video_plane_height // 2
            video_points_offset[:, 1] -= 1.0

            # Project points on marker using cv2.projectPoints
            video_pts_offset_plane, _ = cv2.projectPoints(
                video_points_offset, rvec, tvec, camera_matrix, dist_coeffs)

            # Or project points "manually" as derived in (19)
            video_pts_offset_plane = project_points_manual(
            video_points_offset, rvec, tvec, camera_matrix, dist_coeffs)
            
            video_pts_offset_plane = np.float32(
                video_pts_offset_plane).reshape(-1, 2)

            # Get transfromation matrix using cv2.getPerspectiveTransform
            M_offset = cv2.getPerspectiveTransform(
                src_pts, video_pts_offset_plane)

            # AR like
            warped_video_offset = cv2.warpPerspective(
                video_resized, M_offset, (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, 1, warped_video_offset, 1, 0)

    cv2.imshow("Video on cube plane", frame)
    cv2.waitKey(1)

    if terminate:
        break

cap.release()
cv2.destroyAllWindows()
