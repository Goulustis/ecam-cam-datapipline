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
from format_data.format_utils import EventBuffer, find_clear_val_test
from format_data.eimg_maker import ev_to_eimg
from extrinsics_creator.create_rel_cam import apply_rel_cam, read_rel_cam
from extrinsics_visualization.colmap_scene_manager import ColmapSceneManager



# COLMAP_CAMERA_F="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/calib_new_v4_recons/sparse/0/cameras.bin"
# PROPHESEE_CAM_F="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/intrinsics_prophesee.json"

COLMAP_CAMERA_F="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/rgb-evs-cam-drivers/data_rgb/calib_v7_recons/sparse/0/cameras.bin"
PROPHESEE_CAM_F="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/rgb-evs-cam-drivers/intrinsics/ecam_K_v7.json"
EDNERF_DATA_DIR="/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data"

def load_txt(scale_f):
    with open(scale_f, "r") as f:
        return float(f.read())

def undistort_and_save_img(colmap_dir, save_dir, cond, ret_k_only=False):
    """
    cond (bools)
    returns:
        new_K (np.array[3,3]): rectified intrinsic matrix
        new_img_size (tuple[2,]): rectified image size in (h, w)
    """
    img_fs = sorted(glob.glob(osp.join(colmap_dir, "images", "*.png")))
    
    img_fs = [img_f for (b, img_f) in zip(cond, img_fs) if b]

    cam_bin_f = osp.join(colmap_dir, "sparse/0/cameras.bin")
    K, D = read_colmap_cam_param(cam_bin_f)

    im_h, im_w = cv2.imread(img_fs[0]).shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (im_w, im_h), 1, (im_w, im_h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (im_w, im_h), cv2.CV_32FC1)
    x, y, w, h = roi
    new_K[0, 2] -= x
    new_K[1, 2] -= y

    if ret_k_only:
        return new_K, (h, w)

    def undist_save_fn(inp):
        idx, img_f = inp
        img = cv2.imread(img_f)
        undist_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        undist_img = undist_img[y:y+h, x:x+w]
        save_f = osp.join(save_dir, str(idx).zfill(5) + ".png")
        cv2.imwrite(save_f, undist_img)

    parallel_map(undist_save_fn, list(zip(list(range(len(img_fs))), img_fs)), show_pbar=True, desc="undistorting and saving")
    
    return new_K, (h, w)


def load_st_end_trigs(work_dir):
    st_trig_f = osp.join(work_dir, "st_triggers.txt")
    end_trig_f = osp.join(work_dir, "end_triggers.txt")
    st_trigs, end_trigs = np.loadtxt(st_trig_f), np.loadtxt(end_trig_f)

    trig_f = osp.join(work_dir, "triggers.txt")
    trigs = np.loadtxt(trig_f)
    assert np.abs(trigs[0] - st_trigs[0]) < 1e-8, "trigs need to be same as st_trig because camera trigger noise fix"

    exposure_t = end_trigs[0] - st_trigs[0]
    st_trigs = trigs
    end_trigs = st_trigs + exposure_t

    return st_trigs, end_trigs


def update_poses_with_K(poses, K, img_hw):
    poses = poses.transpose(2,0,1)
    poses = poses[:,:,:4]  # (n, 3, 4)

    fx = K[0,0]
    h, w =  img_hw
    hwfs = np.stack([np.array([h, w, fx])]*len(poses))[..., None]
    new_poses_bounds = np.concatenate([poses, hwfs], axis = -1).transpose(1,2,0)

    return new_poses_bounds



def make_new_poses_bounds(poses, new_K, work_dir, img_hw, cond, n_bins = 4):
    """
    inputs:
        poses (np.ndarray[n, 4,4]): camera matrices
        new_K (np.array(3,3)): camera intrisinc matrix
        img_hw (tuple): size of image (h, 2)
        n_bins (int): number of bins to split rgb exposure time into
    
    returns:
        new_poses_bounds (n_frame*(n_bins - 1), 17): llff camera format
        cam_ts (list): camera times at bin timesteps
        mid_cam_ts (list): camera times at center of exposure of frame
    """
    st_trigs, end_trigs = load_st_end_trigs(work_dir)
    st_trigs, end_trigs = st_trigs[cond], end_trigs[cond]
    mean_ts = (st_trigs + end_trigs) * 0.5

    delta_t = (end_trigs[0] - st_trigs[0])/(n_bins - 1)
    t_steps = delta_t*np.array(list(range(n_bins)))

    cam_ts = st_trigs[..., None] + t_steps[None]
    cam_ts = cam_ts.reshape(-1)
    poses = poses.transpose(2,0,1)
    Rs, Ts = poses[:,:3,:3], poses[:, :3, 3]
    spline = CameraSpline(mean_ts, Rs, Ts)

    interp_T, interp_R = spline.interpolate(cam_ts)
    interp_E = np.concatenate([interp_R, interp_T[..., None]], axis=-1)
    
    fx = new_K[0,0]
    h, w =  img_hw
    hwfs = np.stack([np.array([h, w, fx])]*len(interp_E))[..., None]
    new_poses_bounds = np.concatenate([interp_E, hwfs], axis = -1).transpose(1,2,0)

    return new_poses_bounds, cam_ts, mean_ts



def make_eimgs(K, D, n_bins:int, cam_ts=None, ev_f:str=None, img_size=(720, 1280)):
    """
    cam_ts (list: [t_i ...]) of shape (n_frames * n_bins) for eimgs; will be reshaped to (n_frames, n_bins)
    K (np.array (3,3))

    ev_f: if None, calculate new_K and return 
    """
    print("startingt to make eimg")

    im_h, im_w = img_size
    cam_ts = cam_ts.reshape(-1, n_bins)
    eimgs = np.zeros((len(cam_ts), n_bins - 1, im_h, im_w), dtype=np.int8)

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (im_w, im_h), 1, (im_w, im_h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (im_w, im_h), 5)
    x, y, w, h = roi
    new_K[0, 2] -= x
    new_K[1, 2] -= y

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
    
    return eimgs.reshape(*eimgs.shape[:2], -1), new_K, (h, w)


def write_metadata(save_f, **kwargs):
    with open(save_f, "w") as f:
        json.dump(kwargs, f, indent=2)


def make_dataset_json(colmap_manager: ColmapSceneManager, cond):
    ignore_last = len(colmap_manager.image_ids) - cond.sum()

    all_ids = np.arange(cond.sum()).tolist()
    all_ids = [int(e) for e in all_ids]
    val_ids, test_ids = find_clear_val_test(colmap_manager, ignore_last=ignore_last)
    train_ids = sorted(set(all_ids) - set(val_ids + test_ids))

    dataset_json = {
        "counts": len(all_ids),
        "num_exemplars": len(train_ids),
        "ids": all_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids
    }
    return dataset_json


def sample_points(pts3d:dict, n_pnts = 20000):
    if n_pnts > len(pts3d):
        return pts3d
    
    # Convert dictionary keys to a list and sample n keys using NumPy
    keys = list(pts3d.keys())
    sampled_keys = np.random.choice(keys, n_pnts, replace=False)
    
    # Create a new dictionary with the sampled keys
    sampled_dict = {key: pts3d[key] for key in sampled_keys}
    
    return sampled_dict



def main(work_dir, targ_dir, n_bins = 4, cam_only=False):
    ######################## format rgb #####################
    ev_f = osp.join(work_dir, "processed_events.h5")
    colmap_dir = osp.join(work_dir, osp.basename(work_dir) + "_recon") ## XXX_recons
    colcam_set_dir = osp.join(EDNERF_DATA_DIR, osp.basename(work_dir), "colcam_set")

    save_img_dir = osp.join(targ_dir, "images")
    os.makedirs(save_img_dir, exist_ok=True)
    colmap_manager = ColmapSceneManager(colmap_dir)

    _, trig_end = load_st_end_trigs(work_dir)
    cond = trig_end <= EventBuffer(ev_f).t_f[-1]
    col_cond = colmap_manager.get_found_cond(len(cond))
    cond = col_cond & cond
    cond[np.where(cond)[0][-1]] = False
    new_rgb_K, rec_rgb_size = undistort_and_save_img(colmap_dir, save_img_dir, cond, ret_k_only=cam_only)

    rgb_poses, pts3d, perm = load_colmap_data(colmap_dir)
    pts3d = sample_points(pts3d)

    new_rgb_poses, cam_ts, mid_cam_ts = make_new_poses_bounds(rgb_poses, new_rgb_K, work_dir, 
                                                              rec_rgb_size, cond=cond, n_bins=n_bins)

    mid_rgb_poses = update_poses_with_K(rgb_poses, new_rgb_K, rec_rgb_size)
    save_poses(osp.join(targ_dir, "mid_rgb_poses_bounds.npy"), mid_rgb_poses, pts3d, perm)

    save_f = osp.join(targ_dir, "rgb_poses_bounds.npy")
    save_poses(save_f, new_rgb_poses, pts3d, perm)  # this'll calculate near far plane

    targ_dataset_json_f = osp.join(targ_dir, "dataset.json")
    if not osp.exists(targ_dataset_json_f):
        if osp.exists(osp.join(colcam_set_dir, "dataset.json")):
            shutil.copy(osp.join(colcam_set_dir, "dataset.json"), targ_dir)
        else:
            dataset_json = make_dataset_json(colmap_manager, cond)
            with open(targ_dataset_json_f, "w") as f:
                json.dump(dataset_json, f, indent=2)
    else:
        print(f"dataset json already exists at: {targ_dataset_json_f}")
    #########################################################


    ############### format events ##############
    ev_f = osp.join(work_dir, "processed_events.h5") if not cam_only else None
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
    K_to_list = lambda x : [x[0,0], x[1,1], x[0,2], x[1,2]]
    write_metadata(osp.join(targ_dir, "metadata.json"), evs_hw=eimg_size, n_bins=n_bins, 
                                                        evs_K=K_to_list(new_ecam_K), 
                                                        rgb_K=K_to_list(new_rgb_K),
                                                        K=["fx", "fy", "cx", "cy"],
                                                        n_valid_img=int(cond.sum()),
                                                        mid_cam_ts=mid_cam_ts.tolist(),
                                                        colmap_scale=scale,
                                                        ev_cam_ts=cam_ts.tolist())


    rel_json_f = osp.join(osp.join(work_dir, "rel_cam.json"))
    dst_f = osp.join(targ_dir, "rel_cam.json")
    if not osp.exists(dst_f):
        shutil.copy(rel_json_f, dst_f)

def none_or_int(value):
    if value is None or value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. It should be 'None' or an integer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="")
    parser.add_argument("--targ_dir", type=str, default="")
    parser.add_argument("--n_bins", type=int, default=4)
    parser.add_argument("--delta_t", type=none_or_int, default=None)
    parser.add_argument("--cam_only", type=bool, default=False, help="if true, will process camera only. no events or images will be copied/processed")
    args = parser.parse_args()

    if args.delta_t is not None:
        prev_ts, next_ts = load_st_end_trigs(args.work_dir)
        args.n_bins = int(np.ceil((next_ts[0] - prev_ts[0])/args.delta_t)) + 1
        print("n_bins changed to", args.n_bins)
        print("new time", (next_ts[0] - prev_ts[0])/(args.n_bins - 1))

    main(args.work_dir, args.targ_dir, args.n_bins)
    # main("/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1",
    #     #  "/ubc/cs/research/kmyi/matthew/projects/E2NeRF/data/real-world/atrium_finer",
    #      "debug",
    #      args.n_bins,
    #      cam_only=False)
