import json
import os.path as osp
import glob
import os
import cv2
import numpy as np


def load_json_cam(cam_f):
    with open(cam_f, "r") as f:
        data = json.load(f)
        R, pos = np.array(data["orientation"]), np.array(data["position"])
        t = -(pos@R.T).T
        t = t.reshape(-1,1)
    
    return np.concatenate([R, t], axis=-1)

def load_json_intr(cam_f):
    with open(cam_f, "r") as f:
        data = json.load(f)
        fx = fy = data["focal_length"]
        cx, cy = data["principal_point"]
        k1, k2, k3 = data["radial_distortion"]
        p1, p2 = data["tangential_distortion"]
    
    return np.array([[fx, 0, cx],
                    [0,   fy, cy],
                    [0, 0, 1]]), (k1,k2,p1,p2)


class ColcamSceneManager:

    def __init__(self, data_dir):
        """
        expect: xxx/colcam_set
        """
        self.data_dir = data_dir
        self.dataset_json = osp.join(self.data_dir, "dataset.json")
        self.img_fs = sorted(glob.glob(osp.join(data_dir, "rgb", "1x", "*.png")))

        with open(self.dataset_json, "r") as f:
            data = json.load(f)
            self.img_ids = [int(e) for e in data["ids"]]
    

        self.img_fs = [self.img_fs[e] for e in self.img_ids]

        self.cam_fs = sorted(glob.glob(osp.join(self.data_dir, "camera", "*.json")))
    
    def get_img(self, idx):
        return cv2.imread(self.img_fs[idx])

    def get_intrnxs(self):
        return load_json_intr(self.cam_fs[0])
    

    def get_extrnxs(self, idx):
        return load_json_cam(self.cam_fs[idx])

    def __len__(self):
        return len(self.img_fs)


class EcamSceneManager(ColcamSceneManager):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cam_fs = sorted(glob.glob(osp.join(self.data_dir, "camera", "*.json")))
        self.eimgs = np.load(osp.join(self.data_dir, "eimgs", "eimgs_1x.npy"), "r")

    def get_img(self, idx):
        img = np.stack([(self.eimgs[idx] != 0).astype(np.uint8) * 255]*3, axis=-1)
        return img

    def __len__(self):
        return len(self.eimgs)


MANAGER_DICT = {"colcam_set": ColcamSceneManager,
                "ecam_set": EcamSceneManager}

