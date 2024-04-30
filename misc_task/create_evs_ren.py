import glob
import os.path as osp
import os
import numpy as np 
import cv2
import shutil
from tqdm import tqdm
import json


class EcamScene:
    def __init__(self, ecam_path) -> None:
        self.ecam_path = ecam_path
        self.prev_cam_fs = sorted(glob.glob(osp.join(ecam_path, "prev_camera", "*.json")))
        self.next_cam_fs = sorted(glob.glob(osp.join(ecam_path, "next_camera", "*.json")))
        self.eimgs = np.load(osp.join(ecam_path, "eimgs", "eimgs_1x.npy"), "r")

        self.meta_f = osp.join(ecam_path, "metadata.json")
        self.data_f = osp.join(ecam_path, "dataset.json")
        
        with open(self.meta_f, "r") as f:
            self.meta = json.load(f)

    @property
    def n_frames(self):
        return len(self.prev_cam_fs)

    def get_prev_cam(self, idx):
        return self.prev_cam_fs[idx]
    
    def get_next_cam(self, idx):
        return self.next_cam_fs[idx]
    
    def get_evimg(self, idx):
        return self.eimgs[idx]

    @property
    def colmap_scale(self):
        return self.meta["colmap_scale"]


def create_metadata(scene:EcamScene, n_ren:int, targ_dir:str):
    
    metadata = {
        "colmap_scale": scene.colmap_scale,
    }

    for i in range(n_ren):
        metadata[str(i).zfill(5)] = {
            "warp_id": i,
            "appearance_id": i,
            "camera_id": 0
        }
    
    with open(osp.join(targ_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def create_dataset(n_ren:int, targ_dir:str):

    all_ids = [str(i).zfill(5) for i in range(n_ren)]
    dataset_json = {
        "count": n_ren,
        "num_exemplars": n_ren,
        "ids": all_ids,
        "train_ids": [all_ids[0]],
        "val_ids": all_ids,
        "test_ids": all_ids
    }

    with open(osp.join(targ_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)



def main(scene_path:str, loc:float, targ_dir:str, n_ren:int=64):
    """
    scene_path (str): path to scene
    loc (float): in [0, 1], the location of where to start the sequence
    """
    if targ_dir is None:
        targ_dir = osp.join(osp.dirname(scene_path), osp.basename(scene_path).split('.')[0] + '_evs_ren')
    
    os.makedirs(targ_dir, exist_ok=True)

    ecam_path = osp.join(scene_path, 'ecam_set')
    ecam_scene = EcamScene(ecam_path)
    st_idx = int(ecam_scene.n_frames * loc)
    end_idx = st_idx + n_ren

    targ_cam_dir = osp.join(targ_dir, "camera")
    eimg_dir = osp.join(targ_dir, "rgb", "1x")
    os.makedirs(targ_cam_dir, exist_ok=True)
    os.makedirs(eimg_dir, exist_ok=True)

    dst_cnt = 0
    eimgs = []
    for idx in tqdm(range(st_idx, end_idx), desc="creating evs eval data"):
        dst_idx = 2*dst_cnt
        prev_cam = ecam_scene.get_prev_cam(idx)
        next_cam = ecam_scene.get_next_cam(idx)
        eimg = np.abs(ecam_scene.get_evimg(idx)).astype(np.float32)
        eimgs.append(ecam_scene.get_evimg(idx))

        dst_prev_cam_f = osp.join(targ_cam_dir, f"{str(dst_idx).zfill(5)}.json")
        dst_next_cam_f = osp.join(targ_cam_dir, f"{str(dst_idx+1).zfill(5)}.json")

        os.symlink(prev_cam, dst_prev_cam_f)
        os.symlink(next_cam, dst_next_cam_f)

        eimg_png = ((eimg != 0).astype(np.uint8) * 255)
        cv2.imwrite(osp.join(eimg_dir, f"{str(dst_idx).zfill(5)}.png"), eimg_png)
        cv2.imwrite(osp.join(eimg_dir, f"{str(dst_idx+1).zfill(5)}.png"), eimg_png)

        dst_cnt += 1
    
    
    create_dataset(n_ren, targ_dir)
    create_metadata(ecam_scene, n_ren, targ_dir)
    np.save(osp.join(targ_dir, "eimgs_1x.npy"), np.stack(eimgs, axis=0))
    os.symlink(ecam_path, osp.join(osp.dirname(targ_dir), "ecam_set"))
    os.symlink(osp.join(scene_path, "rel_cam.json"), osp.join(osp.dirname(targ_dir), "rel_cam.json"))



if __name__ == "__main__":
    main(
        "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/atrium_b2_v1_rect",
        0.5,
        "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/atrium_evs_ren/colcam_set"
    )