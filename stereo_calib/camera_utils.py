import cv2
import json
import numpy as np
from viz.proj_pts3d_eimgs import read_colmap_cam
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