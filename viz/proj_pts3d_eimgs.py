import sys
sys.path.append(".")

import numpy as np
import json
from colmap_find_scale.read_write_model import read_points3D_binary, read_cameras_binary, read_images_binary
import cv2
import matplotlib.pyplot as plt
import os
import os.path as osp
import glob


SAVE_DIR = "./tmp/proj_imgs"
os.makedirs(SAVE_DIR, exist_ok=True)
def read_cvcam(cam_path, prefix = 2):
    """
    prefix (int): 1 for color camera; 2 for event camera}
    """
    # read intrinscs, M2, and distortion, d2
    with open(cam_path, "r") as f:
        data = json.load(f)
    
    # dist = 
    return {f"M{prefix}": np.array(data[f"M{prefix}"]), f"d{prefix}": np.array(data[f"dist{prefix}"][:4])}


def read_pts3d(pts3d_path, ret_hom = False, img_id=-1):
    to_hom = lambda x : np.concatenate([x, np.ones((len(x),1))], 1)
    pts3d = read_points3D_binary(pts3d_path)

    xyzs = []
    for k,v in pts3d.items():
        if img_id in v.image_ids or img_id == -1:
            xyzs.append(v.xyz)
    
    if ret_hom:
        return to_hom(np.stack(xyzs))
    else:
        return np.stack(xyzs)

def read_pts3d_from_ids(pts3d_path, pnt_ids, ret_hom = False):
    to_hom = lambda x : np.concatenate([x, np.ones((len(x),1))], 1)
    pts3d = read_points3D_binary(pts3d_path)

    xyzs = []
    for k,v in pts3d.items():
        if v.id in pnt_ids:
            xyzs.append(v.xyz)
    
    if ret_hom:
        return to_hom(np.stack(xyzs))
    else:
        return np.stack(xyzs)


def read_wcam(cam_path):
    # read world camera
    with open(cam_path, "r") as f:
        data = json.load(f)
    
    R = np.array(data['orientation'])
    t = np.array(data['position'])[None]
    t = (-t@R.T).squeeze()
    t = t[..., None]

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

def cv_draw_points(img, prj_pnts):

    for p in prj_pnts:
        cv2.circle(img, tuple(p.astype(int)), radius=2, color=(0,255,0), thickness=-1)
    
    return img


def create_proj_img_v2(pnts, Rw, tw, int_mtx, dist, img, fig_name="ecam_proj", ret_img=False):
    h, w = img.shape[:2]
    prj_pnts, _ = cv2.projectPoints(pnts, Rw, tw, int_mtx, dist)
    prj_pnts = prj_pnts.squeeze()
    keep_cond = (prj_pnts[:, 0] > 0) & (prj_pnts[:,0] < w) & \
                (prj_pnts[:, 1] > 0) & (prj_pnts[:,1] < h)
    prj_pnts = prj_pnts[keep_cond]

    if ret_img:
        return cv_draw_points(img, prj_pnts)
    else:    
        if len(img.shape) == 2 or img.shape[-1] == 1:
            plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            plt.imshow(img)

        plt.scatter(prj_pnts[:,0], prj_pnts[:,1], s=1, c='red')
        plt.savefig(f"{fig_name}.png")


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

def read_colmap_img(path, img_id):
    cam = read_cameras_binary(path)[img_id]
    return cam

def find_most_similar(colmap_img, cam_dir):
    col_R, col_t = colmap_img.qvec2rotmat(), colmap_img.tvec
    cam_fs = sorted(glob.glob(osp.join(cam_dir, "*")))

    min_val = 1000
    sim_f, sim_param = None, None
    for cam_f in cam_fs:
        pro_cam = read_wcam(cam_f)
        R, t = pro_cam[:,:3], pro_cam[:,3:]
        val = np.abs(col_R - R).mean()
        if val < min_val:
            min_val = val
            sim_f, sim_param = cam_f, (R, t)
    
    return sim_f, sim_param


def proj_colcam():
    pts3d_path = 'data/rgb_checker/rgb_checker_recon/sparse/0/points3D.bin'
    colmap_colcam_path = 'data/rgb_checker/rgb_checker_recon/sparse/0/cameras.bin'
    cv_colcam_path = 'data/rgb_checker/rel_cam.json'
    img_path = "data/formatted_rgb_checker/colcam_set/rgb/1x/000241.png"
    wcam_path = "data/formatted_rgb_checker/colcam_set/camera/000241.json"

    img_id = int(osp.basename(img_path).split(".")[0]) + 1
    colmap_img = read_images_binary('data/rgb_checker/rgb_checker_recon/sparse/0/images.bin')[img_id]
    # pts3d = read_pts3d(pts3d_path, False, img_id)
    pts3d = read_pts3d_from_ids(pts3d_path, colmap_img.point3D_ids)
    colmap_colcam = read_colmap_cam(colmap_colcam_path, scale=1)
    cv_colcam = read_cvcam(cv_colcam_path, 1)
    wcam = read_wcam(wcam_path)
    img = plt.imread(img_path)

    cam = cv_colcam
    # create_undist_proj_img(img, cam["M1"], cam["d1"], wcam, pts3d, "colmap_colcam")
    create_proj_img_v2(pts3d, wcam[:,:3], wcam[:,3:].squeeze(), 
                       cam['M1'], cam["d1"], img, "colcam_proj")
    
    # sim_meta = find_most_similar(colmap_img, osp.dirname(wcam_path))
    # create_proj_img_v2(pts3d, colmap_img.qvec2rotmat(), colmap_img.tvec, 
    #                    cam['M1'], cam["d1"], img, "colcam_proj")


def proj_ecam():
    pts3d_path = 'data/rgb_checker/rgb_checker_recon/sparse/0/points3D.bin'
    eimg_path = "data/rgb_checker/events_imgs/0000000008036001000.png"
    cv_cam_path = 'data/rgb_checker/rel_cam.json'
    wcam_path = "data/formatted_rgb_checker/ecam_set/camera/001179.json"

    colmap_img = read_images_binary('data/rgb_checker/rgb_checker_recon/sparse/0/images.bin')[242]
    pts3d = read_pts3d_from_ids(pts3d_path, colmap_img.point3D_ids)
    img = plt.imread(eimg_path)
    cv_ecam = read_cvcam(cv_cam_path, 2)
    wcam = read_wcam(wcam_path)

    create_proj_img_v2(pts3d, wcam[:,:3], wcam[:,3:], cv_ecam["M2"], cv_ecam["d2"], img, "ecam_proj")


if __name__ == "__main__":
    proj_colcam()