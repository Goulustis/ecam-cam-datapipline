import glob
import os
import os.path as osp
import shutil
import json
import numpy as np
from tqdm import tqdm

from stereo_calib.data_scene_manager import ColcamSceneManager, EcamSceneManager
from misc_task.create_prev_next_cams import create_prev_next_cams


def symlink_directory(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for src_file in tqdm(glob.glob(osp.join(src_dir, "*")), desc=f"symlinking, {osp.basename(src_dir)}"):
        src_file_name = osp.basename(src_file)
        dst_file = osp.join(dst_dir, src_file_name)

        if "dataset.json" in src_file_name:
            if not osp.exists(dst_file):
                shutil.copy(src_file, dst_file)
        else:
            if not osp.exists(dst_file):
                os.symlink(src_file, dst_file)


def sparcify_colcam(manager: ColcamSceneManager, targ_colcam_dir, n_train=40):
    symlink_directory(manager.data_dir, targ_colcam_dir)
    
    dataset_f = osp.join(manager.data_dir, "dataset.json")
    with open(dataset_f, "r") as f:
        dataset = json.load(f)

    train_ids = dataset["train_ids"]
    n_skip = len(train_ids) // n_train
    new_train_ids = train_ids[::n_skip]
    dataset["train_ids"] = new_train_ids

    save_dataset_f = osp.join(targ_colcam_dir, "dataset.json")
    with open(save_dataset_f, "w") as f:
        json.dump(dataset, f, indent=2)

    img_ts =  manager.ts
    return img_ts[list(map(int, new_train_ids))]
    

def sparcify_ecam_set(ecam_dir, targ_ecam_dir, train_rgb_ts, t_gap=16000):
    symlink_directory(ecam_dir, targ_ecam_dir)
    os.makedirs(targ_ecam_dir, exist_ok=True)
    prev_cam_dir = osp.join(ecam_dir, "prev_camera")
    if not osp.exists(prev_cam_dir):
        create_prev_next_cams(osp.join(ecam_dir, "camera"))
    
    manager = EcamSceneManager(ecam_dir)
    all_ids = np.arange(len(manager.eimgs))

    train_ids = []
    for rgb_t in train_rgb_ts:
        st_t = rgb_t - t_gap/2
        en_t = rgb_t + t_gap/2
        cond = (manager.ts >= st_t) & (manager.ts <= en_t)
        train_ids.extend(all_ids[cond])

    train_ids = sorted(np.unique(train_ids))
    train_ids = [str(e).zfill(6) for e in train_ids]

    dataset = {
        "count": len(manager.eimgs),
        "num_exemplars": len(train_ids),
        "train_ids" : train_ids,
        "val_ids":[]
    }

    with open(osp.join(targ_ecam_dir, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=2)


def main(scene_dir):
    targ_dir = scene_dir + "_sparse"
    t_gap = 22000 # this should be less than or equal to exposure time or 1/fps

    targ_colcam_dir = osp.join(targ_dir, "colcam_set")
    targ_ecam_dir = osp.join(targ_dir, "ecam_set")

    colcam_dir = osp.join(scene_dir, "colcam_set")
    ecam_dir = osp.join(scene_dir, "ecam_set")

    colManager = ColcamSceneManager(colcam_dir)
    print("sparcifying colcam set")
    train_rgb_ts = sparcify_colcam(colManager, targ_colcam_dir)

    print("sparcifying ecam set")
    t_deltas = np.diff(colManager.ts)
    assert (t_deltas >= 0).all(), "timestamps should be monotonically increasing"
    # t_gap = min(t_deltas.mean() + t_deltas.std(), t_gap)
    # t_gap = t_deltas.mean() + t_deltas.std()

    sparcify_ecam_set(ecam_dir, targ_ecam_dir, train_rgb_ts, t_gap=t_gap)
    print("done")


if __name__ == "__main__":
    scene_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/depth_var_1_lr_000000"
    main(scene_dir)