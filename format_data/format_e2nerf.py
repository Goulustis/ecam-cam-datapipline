import glob
import os
import os.path as osp
import cv2
from functools import partial
import numpy as np
import shutil
from tqdm import tqdm
import torch
import json
import argparse

from utils.misc import parallel_map
from llff.poses.pose_utils import load_colmap_data, save_poses
from stereo_calib.camera_utils import (read_colmap_cam_param, 
                                       read_prosphesee_ecam_param,
                                       poses_to_w2cs_hwf,
                                       w2cs_hwf_to_poses)
from format_data.slerp_qua import CameraSpline
from format_data.format_utils import EventBuffer
from format_data.eimg_maker import ev_to_eimg
from extrinsics_creator.create_rel_cam import apply_rel_cam, read_rel_cam


COLMAP_CAMERA_F="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/calib_checker_recons/sparse/0/cameras.bin"
PROPHESEE_CAM_F="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/intrinsics_prophesee.json"
EDNERF_DATA_DIR="/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data"

def load_txt(scale_f):
    with open(scale_f, "r") as f:
        return float(f.read())

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


def update_poses_with_K(poses, K, img_hw):
    poses = poses.transpose(2,0,1)
    poses = poses[:,:,:4]  # (n, 3, 4)

    fx = K[0,0]
    h, w =  img_hw
    hwfs = np.stack([np.array([h, w, fx])]*len(poses))[..., None]
    new_poses_bounds = np.concatenate([poses, hwfs], axis = -1).transpose(1,2,0)

    return new_poses_bounds



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



def make_eimgs(K, D, n_bins:int, cam_ts=None, ev_f:str=None, img_size=(720, 1280)):
    """
    cam_ts (list: [[st_t1, end_t1], [st_t2, end_t2] ... ]) for eimgs
    K (np.array (3,3))

    ev_f: if None, calculate new_K and return 
    """

    im_h, im_w = img_size
    cam_ts = cam_ts.reshape(-1, n_bins)
    eimgs = np.zeros((len(cam_ts), n_bins - 1, im_h, im_w), dtype=np.int8)

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (im_w, im_h), 1, (im_w, im_h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (im_w, im_h), 5)
    x, y, w, h = roi

    if ev_f is None:
        return None, new_K, (h, w)

    assert cam_ts is not None, "cam_ts required!"
    buffer = EventBuffer(ev_f)

    undist_fn = lambda x : cv2.remap(x, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    n_frames = 0
    for i in tqdm(range(len(eimgs)), desc="making eimgs"):
        n_frames += 1
        prev_t = 0
        for bi in range(n_bins - 1):
            st_t, end_t = cam_ts[i, bi], cam_ts[i, bi+1]
            is_valid = buffer.valid_time(st_t)

            if not is_valid:
                break

            ts, xs, ys, ps = buffer.retrieve_data(st_t, end_t)
            
            eimg = ev_to_eimg(xs, ys, ps)
            
            pos_eimg, neg_eimg = np.copy(eimg), np.copy(eimg)
            pos_cond = eimg > 0
            pos_eimg[~pos_cond] = 0
            neg_eimg[pos_cond] = 0
            pos_eimg, neg_eimg = pos_eimg.astype(np.uint8), np.abs(neg_eimg).astype(np.uint8)
            pos_re, neg_re = undist_fn(pos_eimg), undist_fn(neg_eimg)
            
            eimgs[i, bi] = pos_re.astype(np.int8) + neg_re.astype(np.int8) * -1
            
            buffer.drop_cache_by_t(prev_t)
            prev_t = st_t
            
        if not is_valid:
            break
    
    eimgs = eimgs[:n_frames, :, y:y+h, x:x+w]
    
    # TODO: FIGURE OUT WHY E2NERF FLATTEN THIS
    return eimgs.reshape(*eimgs.shape[:2], -1), new_K, (h, w)


def write_metadata(save_f, **kwargs):
    with open(save_f, "w") as f:
        json.dump(kwargs, f, indent=2)

def main(work_dir, targ_dir, n_bins = 4):
    ######################## format rgb #####################
    ev_f = osp.join(work_dir, "processed_events.h5")
    colmap_dir = osp.join(work_dir, osp.basename(work_dir) + "_recon") ## XXX_recons
    colcam_set_dir = osp.join(EDNERF_DATA_DIR, osp.basename(work_dir), "colcam_set")

    save_img_dir = osp.join(targ_dir, "images")
    os.makedirs(save_img_dir, exist_ok=True)
    _, trig_end = load_st_end_trigs(work_dir)
    cond = trig_end <= EventBuffer(ev_f).t_f[-1]
    new_rgb_K = undistort_and_save_img(colmap_dir, save_img_dir, cond)

    rgb_poses, pts3d, perm = load_colmap_data(colmap_dir)

    img_f = glob.glob(osp.join(save_img_dir, "*.png"))[0]
    new_rgb_poses, cam_ts = make_new_poses_bounds(rgb_poses, new_rgb_K, work_dir, 
                                              cv2.imread(img_f).shape[:2], n_imgs=cond.sum(), n_bins=n_bins)


    save_f = osp.join(targ_dir, "rgb_poses_bounds.npy")
    save_poses(save_f, new_rgb_poses, pts3d, perm)  # this'll calculate near far plane

    if osp.exists(osp.join(colcam_set_dir, "dataset.json")):
        shutil.copy(osp.join(colcam_set_dir, "dataset.json"), targ_dir)
    #########################################################


    ############### format events ##############
    ev_f = osp.join(work_dir, "processed_events.h5")
    ev_K, ev_D = read_prosphesee_ecam_param(PROPHESEE_CAM_F)
    eimgs, new_ecam_K, eimg_size = make_eimgs(ev_K, ev_D, n_bins, cam_ts, ev_f)
    if eimgs is not None:
        torch.save(torch.from_numpy(eimgs), osp.join(targ_dir, "events.pt"))

    ## shift the rgb camera to event camera
    scale = load_txt(osp.join(colmap_dir, "colmap_scale.txt"))
    rel_cam = read_rel_cam(osp.join(work_dir, "rel_cam.json"))
    new_rgb_w2cs, hwfs = poses_to_w2cs_hwf(new_rgb_poses)
    ecam_w2cs = apply_rel_cam(rel_cam, new_rgb_w2cs, scale)
    ecam_poses = w2cs_hwf_to_poses(ecam_w2cs, hwfs)

    ecam_poses = update_poses_with_K(ecam_poses, new_ecam_K, eimg_size)

    save_f = osp.join(targ_dir, "evs_poses_bounds.npy")
    save_poses(save_f, ecam_poses, pts3d, perm)

    ## meta data, write evs img size, write num bins used
    write_metadata(osp.join(targ_dir, "metadata.json"), evs_hw=eimg_size, n_bins=n_bins)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="")
    parser.add_argument("--targ_dir", type=str, default="")
    parser.add_argument("--n_bins", type=int, default=4)
    args = parser.parse_args()

    main(args.work_dir, args.targ_dir, args.n_bins)

