import json
import os
import os.path as osp
import glob
import shutil
import numpy as np

from stereo_calib.data_scene_manager import ColcamSceneManager, EcamSceneManager
from format_data.format_utils import find_clear_val_test


def change_col_dataset(colcam: ColcamSceneManager, test_find:int, unused: list = None, sparce_fac: int = 3):
    """
    test_find: will search for test in the range images [test_find:] Need at least 30 frames
    unused: list of that are unused, expect a range [st_idx, end_idx]
    sparce_fac (int): factor to sparcify train_ids by, (eg. train = tmp_train[::sparce_fac])
    
    """
    if not "unused_ids" in colcam.dataset:
        test_find = test_find if test_find > 0 else len(colcam) + test_find
        test_ids, val_ids = find_clear_val_test(colcam, ignore_first=test_find, ignore_last=1)
        
        all_ids = set(colcam.image_ids)
        unused_ids = set(range(unused[0], unused[1])) | set(colcam.image_ids[test_find:])
        # train_ids = all_ids - unused_ids - set(val_ids) - set(test_ids)
        tmp_train_ids = all_ids - unused_ids - set(val_ids) - set(test_ids)
        train_ids = set(list(tmp_train_ids)[::sparce_fac])
        unused_ids = unused_ids | (tmp_train_ids - train_ids)

        ori_dataset_f = colcam.dataset_json_f
        if not osp.join(colcam.data_dir, "ori_dataset.json"):
            shutil.copy(ori_dataset_f, osp.join(colcam.data_dir, "ori_dataset.json"))

        new_dataset = { "counts" : len(all_ids),
                        "num_exemplars": len(train_ids),
                        "ids": list(all_ids),
                        "train_ids": list(train_ids),
                        "unused_ids": list(unused_ids),
                        "val_ids": val_ids,
                        "test_ids": test_ids
        }

        with open(ori_dataset_f, "w") as f:
            json.dump(new_dataset, f, indent=2)
    else:
        train_ids = colcam.dataset["train_ids"]

    img_ts = colcam.ts
    return img_ts[sorted(list(map(int, train_ids)))]

# def change_col_dataset(colcam: ColcamSceneManager, test_find:int, unused: list = None):
#     """
#     test_find: will search for test in the range images [test_find:] Need at least 30 frames
#     unused: list of that are unused, expect a range [st_idx, end_idx]
    
#     """
#     test_find = test_find if test_find > 0 else len(colcam) + test_find
#     test_ids, val_ids = colcam.dataset["test_ids"], colcam.dataset["val_ids"]
    
#     all_ids = set(colcam.image_ids)
#     unused_ids = set(colcam.dataset["unused_ids"])
#     tmp_train_ids = all_ids - unused_ids - set(val_ids) - set(test_ids)
#     train_ids = list(tmp_train_ids)[::4]
#     unused_ids = unused_ids | (tmp_train_ids - set(train_ids))

#     ori_dataset_f = colcam.dataset_json_f
#     if not osp.join(colcam.data_dir, "ori_dataset.json"):
#         shutil.copy(ori_dataset_f, osp.join(colcam.data_dir, "ori_dataset.json"))

#     new_dataset = { "counts" : len(all_ids),
#                     "num_exemplars": len(train_ids),
#                     "ids": list(all_ids),
#                     "train_ids": list(train_ids),
#                     "unused_ids": list(unused_ids),
#                     "val_ids": val_ids,
#                     "test_ids": test_ids
#     }

#     with open(ori_dataset_f, "w") as f:
#         json.dump(new_dataset, f, indent=2)

#     img_ts = colcam.ts
#     return img_ts[sorted(list(map(int, train_ids)))]



def change_evs_dataset(manager:EcamSceneManager, train_rgb_ts, t_gap=22000):
    """
    t_gap (int): time in mus, should be less than 1/fps or exposure time
    """
    
    all_ids = np.arange(len(manager.eimgs))
    eimg_ts = manager.ts[:len(all_ids)]

    train_ids = []
    for rgb_t in train_rgb_ts:
        st_t = rgb_t - t_gap/2
        en_t = rgb_t + t_gap/2
        cond = (eimg_ts >= st_t) & (eimg_ts <= en_t)
        train_ids.extend(all_ids[cond])

    train_ids = sorted(np.unique(train_ids))
    train_ids = [str(e).zfill(6) for e in train_ids]

    dataset = {
        "count": len(manager.eimgs),
        "num_exemplars": len(train_ids),
        "train_ids" : train_ids,
        "val_ids":[]
    }

    ori_dataset_f = manager.dataset_json_f
    if not osp.join(manager.data_dir, "ori_dataset.json"):
        shutil.copy(ori_dataset_f, osp.join(manager.data_dir, "ori_dataset.json"))

    with open(osp.join(manager.data_dir, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=2)


def main():
    scene_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/lab_c1"
    colcam_dir = osp.join(scene_dir, "colcam_set")
    ecam_dir = osp.join(scene_dir, "ecam_set")

    colcam = ColcamSceneManager(colcam_dir)
    ecam = EcamSceneManager(ecam_dir)

    print("updating colcam dataset")
    train_rgb_ts = change_col_dataset(colcam, test_find=-30*6, unused=[0, 1])
    change_evs_dataset(ecam, train_rgb_ts, t_gap=30000)


if __name__ == "__main__":
    main()