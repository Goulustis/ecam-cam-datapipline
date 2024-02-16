import cv2
import json
import numpy as np
from colmap_find_scale.read_write_model import read_points3D_binary, read_cameras_binary, read_images_binary

# reference:
# https://github.com/colmap/colmap/blob/8e7093d22b324ce71c70d368d852f0ad91743808/src/colmap/sensor/models.h#L268C34-L268C34
def read_colmap_cam(cam_path, scale=1):
    # fx, fy, cx, cy, k1, k2, k3, k4
    cam = read_cameras_binary(cam_path)
    cam = cam[1]

    fx, fy, cx, cy = cam.params[:4]*scale
    int_mtx = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]]) 
    
    # cv2 model fx, fy, cx, cy, k1, k2, p1, p2
    return {"M1": int_mtx, "d1":cam.params[4:]}

def read_colmap_cam_param(cam_path):
    out = read_colmap_cam(cam_path, scale=1)

    return out["M1"], out["d1"]

def read_prosphesee_ecam_param(cam_path):
    
    with open(cam_path,"r") as f:
        data = json.load(f)
    
    return np.array(data["camera_matrix"]['data']).reshape(3,3), \
           np.array(data['distortion_coefficients']['data'])


def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Undistorts an image using the given camera matrix and distortion coefficients.

    Parameters:
    - image: The distorted input image.
    - camera_matrix: The camera intrinsic matrix.
    - dist_coeffs: The distortion coefficients (k1, k2, p1, p2).

    Returns:
    - undistorted_image: The undistorted output image.
    """

    # Get the image size
    h, w = image.shape[:2]

    # Calculate the undistortion and rectification transformation map
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5)

    # Remap the original image to the new undistorted image
    undist_img = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # Crop the image to the ROI
    x, y, w, h = roi
    undist_img = undist_img[y:y+h, x:x+w]

    return undist_img



if __name__ == "__main__":
    colcam_path = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/calib_checker_recons/sparse/0/cameras.bin"
    img_f = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/black_seoul_b3_v3/colcam_set/rgb/1x/00000.png"
    K, D = read_colmap_cam_param(colcam_path)
    img = cv2.imread(img_f)
    undist = undistort_image(img, K, D)
    
