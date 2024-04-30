import json
import numpy as np
import cv2
import os
import os.path as osp
import glob
from tqdm import tqdm

from stereo_calib.data_scene_manager import EdEvimoManager
from stereo_calib.camera_utils import w2cs_hwf_to_poses, to_homogenous
from utils.misc import parallel_map


def undistort_and_save_img(sceneManager:EdEvimoManager, save_dir):

    K, D = sceneManager.get_intrnxs()
    im_h, im_w = sceneManager.img_shape
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (im_w, im_h), 1, (im_w, im_h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (im_w, im_h), cv2.CV_32FC1)
    x, y, w, h = roi
    new_K[0, 2] -= x
    new_K[1, 2] -= y

    def undist_save_fn(inp):
        idx, img_f = inp
        img = cv2.imread(img_f)
        undist_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        undist_img = undist_img[y:y+h, x:x+w]
        save_f = osp.join(save_dir, str(idx).zfill(5) + ".png")
        cv2.imwrite(save_f, undist_img)

    parallel_map(undist_save_fn, list(zip(list(range(len(sceneManager))), sceneManager.img_fs)), show_pbar=True, desc="undistorting and saving")
    
    return new_K, (h, w)


def scale_and_save_imgs(img_dir, scale=4):
    targ_dir = osp.join(osp.dirname(img_dir), f"images_{scale}")
    os.makedirs(targ_dir, exist_ok=True)

    img_fs = glob.glob(osp.join(img_dir, "*"))
    
    h, w = cv2.imread(img_fs[0]).shape[:2]

    def resize_fn(img_f):
        targ_f = osp.join(targ_dir, osp.basename(img_f))
        img = cv2.imread(img_f)
        img = cv2.resize(img, (w//scale, h//scale))
        cv2.imwrite(targ_f, img)
    
    parallel_map(resize_fn, img_fs, show_pbar=True, desc="resizing imgs")



def format_and_save_poses_bounds(dst_f, manager: EdEvimoManager, K, out_img_size):
    w2cs = to_homogenous(manager.get_all_extrnxs())
    h, w = out_img_size
    hwf = np.array([h, w, K[0,0]]).reshape(3, 1)
    hwfs = np.tile(hwf[:, np.newaxis], [1, 1, len(w2cs)])

    poses_bounds = w2cs_hwf_to_poses(w2cs, hwfs).transpose(2,0,1).reshape(-1, 15)
    poses_bounds = np.concatenate([poses_bounds, manager.get_all_depths()], axis=1)

    np.save(dst_f, poses_bounds)


def write_metadata(dst_f, K):
    metadata = {"K_keys":["fx", "fy", "cx", "cy"]}
    metadata["K"] = [K[0,0], K[1,1], K[0, 2], K[1, 2]]

    with open(dst_f, "w") as f:
        json.dump(metadata, f, indent=2)


def write_dataset(dst_f, sceneManager:EdEvimoManager):
    dataset = sceneManager.dataset
    train = [int(e) for e in sorted(dataset["train_ids"])]
    val = [int(e) for e in sorted(dataset["val_ids"])]

    dataset_json = {
        "train" : train,
        "val" : val
    }

    with open(dst_f, "w") as f:
        json.dump(dataset_json, f, indent=2)



# def main(src_dir, targ_dir):
def main():
    src_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/depth_var_1_lr_000000/colcam_set"
    targ_dir = "/ubc/cs/research/kmyi/matthew/projects/PDRF/data/depth_var_1_lr_000000"
    os.makedirs(targ_dir, exist_ok=True)

    rgb_save_dir = osp.join(targ_dir, "images")
    os.makedirs(rgb_save_dir, exist_ok=True)

    manager = EdEvimoManager(src_dir)
    new_K, out_img_size = undistort_and_save_img(manager, rgb_save_dir)
    scale_and_save_imgs(rgb_save_dir)

    format_and_save_poses_bounds(osp.join(targ_dir, "poses_bounds.npy"), manager, new_K, out_img_size)

    meta_f = osp.join(targ_dir, "metadata.json")
    write_metadata(meta_f, new_K)

    dataset_f = osp.join(targ_dir, "dataset.json")
    write_dataset(dataset_f, manager)

if __name__ == "__main__":
    main()