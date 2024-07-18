import numpy as np
import os
import os.path as osp
import json
import cv2
import shutil
from tqdm import tqdm

from stereo_calib.data_scene_manager import ColcamSceneManager
from stereo_calib.camera_utils import to_homogenous, inv_mtxs, make_camera_json


def get_max_camera_diff(c2ws):
    camera_positions = c2ws[:,:3,3] 
    
    # Compute the pairwise differences between all camera positions
    diff = camera_positions[:, np.newaxis, :] - camera_positions[np.newaxis, :, :]
    
    # Calculate the Euclidean distance from the pairwise differences
    distances = np.linalg.norm(diff, axis=-1)
    
    # Find the maximum distance, ignoring the diagonal (distances of cameras from themselves)
    neg_inf = np.eye(distances.shape[0]) * -np.inf
    neg_inf[np.isnan(neg_inf)] = 0
    max_dist = np.max(distances + neg_inf)

    return max_dist

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def make_spiral_path(c2w, up, rads, focal, zdelta, zrate, rots, N):
    spiral_c2ws = []
    rads = np.array(list(rads) + [1.])
    
    for theta in tqdm(np.linspace(0., 2. * np.pi * rots, N+1)[:-1], desc="Making spiral path"):
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        spiral_c2ws.append(viewmatrix(z, up, c))
    return np.stack(spiral_c2ws)


def poses_avg(c2ws):

    center = c2ws[:, :3, 3].mean(0)
    vec2 = normalize(c2ws[:, :3, 2].sum(0))
    up = c2ws[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    
    return c2w


def make_spiral(c2ws:np.ndarray, n_views=64, n_rots=2, spiral_focal=None):

    avg_c2w = c2ws[0] #poses_avg(c2ws)

    up = normalize(c2ws[:,:3,1].sum(axis=0))

    if spiral_focal is None:
        spiral_focal = get_max_camera_diff(c2ws)*2
    close_depth = 0.1

    shrink_factor = .8
    zdelta = close_depth * (1 - shrink_factor)
    tt = c2ws[:,:3,3]
    rads = np.percentile(np.abs(tt), 90, 0)

    spiral_path = make_spiral_path(avg_c2w, up, rads, spiral_focal, zdelta, zrate=0.5, rots=n_rots, N=n_views)
    return spiral_path




def create_and_save_spiral(scene: ColcamSceneManager, targ_dir, n_views):
    w2cs = scene.get_all_extrnxs()
    c2ws = inv_mtxs(w2cs)[:,:3,:4]

    K, D = scene.get_intrnxs()
    f = K[0,0]

    h, w = scene.get_img(0).shape[:2]

    spiral_c2ws = make_spiral(c2ws, n_views=n_views)
    spiral_w2cs = inv_mtxs(spiral_c2ws)

    os.makedirs(targ_dir, exist_ok=True)

    intrxs, D = scene.get_intrnxs()
    for i, c2w in enumerate(spiral_w2cs):
        cam_json = make_camera_json(c2w, intrxs, D, (w,h))
        with open(osp.join(targ_dir, str(i).zfill(6) + ".json"), "w") as f:
            json.dump(cam_json, f, indent=2)
    
    return w2cs


def create_and_save_dummy_frames(scene: ColcamSceneManager, targ_dir, n_dummies):
    os.makedirs(targ_dir, exist_ok=True)

    dummy_img = np.full(scene.get_img(0).shape, 128, dtype=np.uint8)
    targ_dummy_f = osp.join(targ_dir, "00000.png")
    cv2.imwrite(targ_dummy_f, dummy_img)

    for i in tqdm(range(1, n_dummies), desc="making dummy frames"):
        save_f = osp.join(targ_dir, str(i).zfill(5) + ".png")

        if not osp.exists(save_f):
            os.symlink(targ_dummy_f, save_f)



def create_and_save_dataset(n_frames, src_dataset_f, targ_dir, n_ren=64):

    with open(src_dataset_f, "r") as f:
        dataset = json.load(f)
    n_train = len(dataset["train_ids"])
    
    targ_dataset_f = osp.join(targ_dir, "dataset.json")
    ids = [str(i).zfill(6) for i in range(n_frames)]

    n_frames_per_ren = n_frames // n_ren
    dataset = {
        "count":n_frames,
        "num_exemplars":n_frames,
        "train_ids":ids[:n_train],
        "val_ids":ids[::n_frames_per_ren],
        "test_ids": []
    }

    with open(targ_dataset_f, "w") as f:
        json.dump(dataset, f, indent=2)



def create_spiral_render_data(src_dir:str, targ_dir:str = None):
    """
    src_dir (str): <scene>/colcam_set
    targ_dir (str): <scene>/<xxx>_colcam_set
    """
    if targ_dir is None:
        targ_dir = osp.join(osp.dirname(src_dir), "spiral_colcam_set")

    src_dataset_f = osp.join(src_dir, "dataset.json")
    scene = ColcamSceneManager(src_dir)
    
    spiral_Ms = create_and_save_spiral(scene, osp.join(targ_dir, "camera"), n_views=len(scene))
    create_and_save_dummy_frames(scene, osp.join(targ_dir, "rgb/1x"), len(spiral_Ms))
    create_and_save_dataset(len(spiral_Ms), src_dataset_f, targ_dir, n_ren=86)

    src_meta_f = osp.join(src_dir, "metadata.json")
    targ_meta_f = osp.join(targ_dir, "metadata.json")

    shutil.copy(src_meta_f, targ_meta_f)



if __name__ == "__main__":
    src_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/daniel_v9/colcam_set"

    create_spiral_render_data(src_dir)