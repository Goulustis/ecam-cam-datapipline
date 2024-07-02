import os
import os.path as osp
import glob
import shutil
from tqdm import tqdm
import json 

def create_prev_next_cams(cam_dir):
    prev_cam_dir = osp.join(osp.dirname(cam_dir), "prev_camera")
    next_cam_dir = osp.join(osp.dirname(cam_dir), "next_camera")

    os.makedirs(prev_cam_dir, exist_ok=True), os.makedirs(next_cam_dir, exist_ok=True)

    src_cam_fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))

    prev_cam_fs = src_cam_fs[:-1]
    next_cam_fs = src_cam_fs[1:]

    for prev_cam_f in tqdm(prev_cam_fs, desc="copying prev cam"):
        prev_cam_f_name = osp.basename(prev_cam_f)
        prev_cam_f_dst = osp.join(prev_cam_dir, prev_cam_f_name)
        shutil.copy(prev_cam_f, prev_cam_f_dst)
    
    for next_cam_f in tqdm(next_cam_fs, desc="copying next cam"):
        next_cam_f_name = osp.basename(next_cam_f)
        next_cam_f_dst = osp.join(next_cam_dir, next_cam_f_name)
        shutil.copy(next_cam_f, next_cam_f_dst)


def update_train_ids(scene_dir):
    colcam_dir = osp.join(scene_dir, "colcam_set")
    dataset_f = osp.join(colcam_dir, "dataset.json")

    with open(dataset_f, "r") as f:
        dataset = json.load(f)
    
    train_ids = dataset["train_ids"]
    train_ids = sorted(list(set(train_ids) - set(dataset["val_ids"])))
    dataset["num_exemplars"] = len(train_ids)
    with open(dataset_f, "w") as f:
        dataset["train_ids"] = train_ids
        json.dump(dataset, f, indent=2)



if __name__ == "__main__":
    # scene_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/halloween_b2_v1_rect"
    # update_train_ids(scene_dir)

    # scene_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/atrium_b2_v1_rect"
    # update_train_ids(scene_dir)

    cam_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/depth_var_1_lr_000000/ecam_set/camera"
    create_prev_next_cams(cam_dir)
