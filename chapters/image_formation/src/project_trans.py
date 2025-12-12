import cv2
import numpy as np

def project_points_manual(pts_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Projects 3D points onto a 2D image plane using a given camera model.

    Args:
        pts_3d (numpy.ndarray): Array of 3D points (N x 3).
        rvec (numpy.ndarray): Rotation vector (3 x 1) in Rodrigues' format.
        tvec (numpy.ndarray): Translation vector (3 x 1).
        camera_matrix (numpy.ndarray): Intrinsic camera matrix (3 x 3).
        dist_coeffs (numpy.ndarray): Distortion coefficients (1 x 5 or 1 x 4).

    Returns:
        numpy.ndarray: Projected 2D points (N x 2).
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Camera Intrinsic Parameters
    f_x, f_y = camera_matrix[0, 0], camera_matrix[1, 1]  # Focal length
    c_x, c_y = camera_matrix[0, 2], camera_matrix[1, 2]  # Principal point
    gamma = camera_matrix[0, 1]  # Skew factor (typically 0)

    # Distortion coefficients
    k1, k2, p1, p2, k3 = dist_coeffs.ravel()

    projected_pts = []
    for pt in pts_3d:
        # Convert 3D point to camera coordinate system
        pt_cam_hat = np.dot(R, pt) + tvec.ravel()
        
        # Points in camera coordinate system (from homogeneous to Euclidean space)
        x_c = pt_cam_hat[0] / pt_cam_hat[2]
        y_c = pt_cam_hat[1] / pt_cam_hat[2]

        # Apply radial and tangential distortion
        r2 = x_c**2 + y_c**2
        
        L_r_x = x_c * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) 
        L_r_y = y_c * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) 
        
        L_t_x = 2 * p1 * x_c * y_c + p2 * (r2 + 2 * x_c**2)
        L_t_y = p1 * (r2 + 2 * y_c**2) + 2 * p2 * x_c * y_c
        
        L_x = L_r_x + L_t_x 
        L_y = L_r_y + L_t_y 
        
        x_d = L_x
        y_d = L_y

        # Convert to pixel coordinates
        x_s = f_x * x_d + gamma * y_d + c_x
        y_s = f_y * y_d + c_y
                
        projected_pts.append((x_s, y_s))

    return np.array(projected_pts, dtype=np.float32)
