import numpy as np
import os.path as osp
import os
import glob
import json
import cv2

class Camera:
    def __init__(self, rel_cam_f):
        self.cam_f = rel_cam_f
        with open(self.cam_f, "r") as f:
            data = json.load(f)
        

        self.w, self.h = 1280, 720
        self.intrxs = np.array(data["M2"])
        self.dist_coeffs = np.array(data["dist2"]).reshape(-1)
    
    def get_dist_coeffs(self):
        return self.dist_coeffs

class EvSceneManager:

    def __init__(self, img_dir, ecam_f, rel_cam_f):
        self.img_fs = sorted(glob.glob(osp.join(img_dir, "*.png")))
        self.ecams = np.load(ecam_f)[:len(self.img_fs)]
        self.camera = Camera(rel_cam_f)
    

    def get_data(self, idx):
        img = cv2.imread(self.img_fs[idx])
        extrinsics = self.ecams[idx]
        return img, self.camera.intrxs, self.ecams[idx], self.camera.dist_coeffs
    

    def __len__(self):
        return len(self.img_fs)
