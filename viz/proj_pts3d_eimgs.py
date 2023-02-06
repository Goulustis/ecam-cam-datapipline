import sys
sys.path.append(".")

import numpy as np
import json
from colmap_find_scale.read_write_model import read_points3D_binary, read_cameras_binary
import cv2
import matplotlib.pyplot as plt

def read_cvcam(cam_path, prefix = 2):
    """
    prefix (int): 1 for color camera; 2 for event camera}
    """
    # read intrinscs, M2, and distortion, d2
    with open(cam_path, "r") as f:
        data = json.load(f)
    
    # dist = 
    return {f"M{prefix}": np.array(data[f"M{prefix}"]), f"d{prefix}": np.array(data[f"dist{prefix}"][:4])}


def read_pts3d(pts3d_path):
    to_hom = lambda x : np.concatenate([x, np.ones((len(x),1))], 1)
    pts3d = read_points3D_binary(pts3d_path)

    xyzs = []
    for k,v in pts3d.items():
        xyzs.append(v.xyz)
    
    return to_hom(np.stack(xyzs))


def read_wcam(wecam_path):
    # read world camera
    with open(wecam_path, "r") as f:
        data = json.load(f)
    
    R = np.array(data['orientation'])
    t = np.array(data['position'])[None]
    t = (-t@R).T

    return np.concatenate([R, t], axis=1)


def create_undist_proj_img(img, intrinsics, distortion, extrinsics, pts3d, cam="ecam"):
    h, w = img.shape[:2]
    optim_cam, roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion,(w,h), 1, (w, h))
    undist_img = cv2.undistort(img, intrinsics, distortion, None, optim_cam)
    x, y, w, h = roi
    undist_img = undist_img[y:y+h, x:x+w]
    # optim_cam = intrinsics
    # undist_img = img

    pix_loc = (optim_cam@extrinsics@pts3d.T).T
    pix_loc = pix_loc[:,:2]/pix_loc[:,2,None]
    
    keep_cond = (pix_loc[:, 0] >= 0) & (pix_loc[:,0] <= w) & \
                (pix_loc[:, 1] >= 0) & (pix_loc[:,1] <= h)
    pix_loc = pix_loc[keep_cond]

    plt.imsave(f"undist_{cam}.png", undist_img)

    plt.imshow(undist_img, cmap='gray', vmin=0, vmax=1)
    plt.scatter(pix_loc[:,0], pix_loc[:,1], s=1)

    plt.savefig(f'{cam}_proj.png')


def read_colmap_cam(cam_path, scale=2):
    # fx, fy, cx, cy, k1, k2, k3, k4
    cam = read_cameras_binary(cam_path)
    cam = cam[1]

    fx, fy, cx, cy = cam.params[:4]*scale
    int_mtx = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]]) 
    
    # cv2 model fx, fy, cx, cy, k1, k2, p1, p2
    return {"M1": int_mtx, "d1":cam.params[4:]}



def create_undist_ecam():
    pts3d_path = 'data/checker/checker_recon/sparse/0/points3D.bin'
    ecam_path = 'data/checker/rel_cam.json'
    wecam_path = 'data/formatted_checkers/ecam_set/camera/000456.json'
    eimg_path = 'data/checker/events_imgs/0000000003865087000.png'


    pts3d = read_pts3d(pts3d_path)
    ecam = read_cvcam(ecam_path, 2)
    wecam = read_wcam(wecam_path)
    eimg = plt.imread(eimg_path)

    create_undist_proj_img(eimg, ecam["M2"], ecam["d2"], wecam, pts3d, "ecam")


def create_undist_colcam():
    pts3d_path = 'data/checker/checker_recon/sparse/0/points3D.bin'
    colmap_colcam_path = 'data/checker/checker_recon/sparse/0/cameras.bin'
    cv_colcam_path = 'data/checker/rel_cam.json'
    img_path = "data/formatted_checkers/colcam_set/rgb/1x/000242.png"
    wcam_path = "data/formatted_checkers/colcam_set/camera/000242.json"

    pts3d = read_pts3d(pts3d_path)
    colmap_colcam = read_colmap_cam(colmap_colcam_path)
    cv_colcam = read_cvcam(cv_colcam_path, 1)
    wcam = read_wcam(wcam_path)
    img = plt.imread(img_path)

    cam = colmap_colcam
    create_undist_proj_img(img, cam["M1"], cam["d1"], wcam, pts3d, "colmap_colcam")


if __name__ == "__main__":
    create_undist_colcam()


