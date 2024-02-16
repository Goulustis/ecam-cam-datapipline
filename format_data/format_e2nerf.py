import glob
import os
import os.path as osp
import cv2
from functools import partial
import numpy as np
import shutil
from tqdm import tqdm
import torch

from utils.misc import parallel_map
from llff.poses.pose_utils import load_colmap_data, save_poses
from stereo_calib.camera_utils import (read_colmap_cam_param, 
                                       read_prosphesee_ecam_param,
                                       poses_to_w2cs_hwf,
                                       w2cs_hwf_to_poses)
from format_data.slerp_qua import CameraSpline
from format_data.format_utils import EventBuffer
from format_data.eimg_maker import ev_to_eimg


COLMAP_CAMERA_F="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/calib_checker_recons/sparse/0/cameras.bin"
PROPHESEE_CAM_F="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/intrinsics_prophesee.json"


def undistort_and_save_img(colmap_dir, save_dir, cond):
    """
    cond (bools)
    """
    img_fs = sorted(glob.glob(osp.join(colmap_dir, "images", "*.png")))
    cam_bin_f = osp.join(colmap_dir, "sparse/0/cameras.bin")
    K, D = read_colmap_cam_param(cam_bin_f)

    im_h, im_w = cv2.imread(img_fs[0]).shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (im_w, im_h), 1, (im_w, im_h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (im_w, im_h), 5)
    x, y, w, h = roi

    def undist_save_fn(inp):
        idx, img_f, cnd = inp
        if cnd:
            img = cv2.imread(img_f)
            undist_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            undist_img = undist_img[y:y+h, x:x+w]
            save_f = osp.join(save_dir, str(idx).zfill(5) + ".png")
            cv2.imwrite(save_f, undist_img)

    parallel_map(undist_save_fn, list(zip(list(range(len(img_fs))), img_fs, cond)), show_pbar=True, desc="undistorting and saving")
    
    return new_K


def load_st_end_trigs(work_dir):
    st_trig_f = osp.join(work_dir, "st_triggers.txt")
    end_trig_f = osp.join(work_dir, "end_triggers.txt")
    st_trigs, end_trigs = np.loadtxt(st_trig_f), np.loadtxt(end_trig_f)
    return st_trigs, end_trigs



def make_new_poses_bounds(poses, new_K, work_dir, img_hw, n_imgs, n_bins = 4):
    """
    n_bins = number of bins as in 
    """
    st_trigs, end_trigs = load_st_end_trigs(work_dir)
    mean_ts = (st_trigs + end_trigs) * 0.5
    mean_ts = mean_ts[:n_imgs]

    delta_t = (end_trigs[0] - st_trigs[0])/(n_bins - 1)
    t_steps = delta_t*np.array(list(range(n_bins)))

    cam_ts = st_trigs[..., None] + t_steps[None]
    cam_ts = cam_ts.reshape(-1)
    poses = poses.transpose(2,0,1)
    Rs, Ts = poses[:,:3,:3], poses[:, :3, 2]
    spline = CameraSpline(mean_ts, Rs, Ts)

    interp_T, interp_R = spline.interpolate(cam_ts)
    interp_E = np.concatenate([interp_R, interp_T[..., None]], axis=-1)
    
    fx = new_K[0,0]
    h, w =  img_hw
    hwfs = np.stack([np.array([h, w, fx])]*len(interp_E))[..., None]
    new_poses_bounds = np.concatenate([interp_E, hwfs], axis = -1).transpose(1,2,0)

    return new_poses_bounds, cam_ts



def make_eimgs(ev_f, cam_ts, K, D, n_bins, img_size=(720, 1280)):
    #TODO: undistort stuff

    buffer = EventBuffer(ev_f)
    im_h, im_w = img_size
    cam_ts = cam_ts.reshape(-1, n_bins)
    eimgs = np.zeros((len(cam_ts), n_bins, im_h, im_w), dtype=np.int8)

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (im_w, im_h), 1, (im_w, im_h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (im_w, im_h), 5)
    x, y, w, h = roi

    undist_fn = lambda x : cv2.remap(x, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    for i in tqdm(range(len(eimgs)), desc="making eimgs"):
        for bi in range(n_bins - 1):
            st_t, end_t = cam_ts[i, bi], cam_ts[i, bi+1]
            ts, xs, ys, ps = buffer.retrieve_data(st_t, end_t)
            
            eimg = ev_to_eimg(xs, ys, ps)
            
            pos_eimg, neg_eimg = np.copy(eimg), np.copy(eimg)
            pos_cond = eimg > 0
            pos_eimg[~pos_cond] = 0
            neg_eimg[pos_cond] = 0
            pos_eimg, neg_eimg = pos_eimg.astype(np.uint8), np.abs(neg_eimg).astype(np.uint8)
            pos_re, neg_re = undist_fn(pos_eimg), undist_fn(neg_eimg)
            
            eimgs[i, bi] = pos_re.astype(np.int8) + neg_re.astype(np.int8) * -1
    
    eimgs = eimgs[:, :, y:y+h, x:x+w]
    
    return eimgs, new_K



def main():
    ######################## format rgb #####################
    work_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/boardroom_b2_v1"
    targ_dir= "e2nerf_dev_boardroom_b2_v1"
    ev_f = osp.join(work_dir, "processed_events.h5")
    n_bins = 4

    colmap_dir = osp.join(work_dir, osp.basename(work_dir) + "_recon") ## XXX_recons
    colcam_set_dir = osp.join("/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data", osp.basename(work_dir), "colcam_set")

    save_img_dir = osp.join(targ_dir, "images")
    os.makedirs(save_img_dir, exist_ok=True)
    _, trig_end = load_st_end_trigs(work_dir)
    cond = trig_end <= EventBuffer(ev_f).t_f[-1]
    new_rgb_K = undistort_and_save_img(colmap_dir, save_img_dir, cond)

    poses, pts3d, perm = load_colmap_data(colmap_dir)

    img_f = glob.glob(osp.join(save_img_dir, "*.png"))[0]
    new_poses, cam_ts = make_new_poses_bounds(poses, new_rgb_K, work_dir, 
                                              cv2.imread(img_f).shape[:2], n_imgs=cond.sum(), n_bins=n_bins)

    save_rgb_pose_dir = osp.join(targ_dir, "rgb_poses")
    os.makedirs(save_rgb_pose_dir, exist_ok=True)
    save_poses(save_rgb_pose_dir, new_poses, pts3d, perm)  # this'll calculate near far plane
    shutil.copy(osp.join(colcam_set_dir, "dataset.json"), save_rgb_pose_dir)
    #########################################################

    ev_f = osp.join(work_dir, "processed_events.h5")
    ev_K, ev_D = read_prosphesee_ecam_param(PROPHESEE_CAM_F)
    eimgs, new_evs_K = make_eimgs(ev_f, cam_ts, ev_K, ev_D, n_bins)
    torch.save(torch.from_numpy(eimgs), osp.join(targ_dir, "events.pt"))




if __name__ == "__main__":
    main()

