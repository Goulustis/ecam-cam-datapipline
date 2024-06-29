import json
import os.path as osp
import glob
import os
import cv2
import numpy as np


def load_json_cam(cam_f):
    """
    returns 3x4
    """
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
        self.dataset_json_f = osp.join(self.data_dir, "dataset.json")
        self.img_fs = sorted(glob.glob(osp.join(data_dir, "rgb", "1x", "*.png")))

        with open(self.dataset_json_f, "r") as f:
            self.dataset = json.load(f)
            self.img_ids = [int(e) for e in self.dataset["ids"]]
            self.train_ids = self.dataset["train_ids"]
            self.val_ids = self.dataset["val_ids"]
    

        self.img_fs = [self.img_fs[e] for e in self.img_ids if e < len(self.img_fs)]

        self.cam_fs = sorted(glob.glob(osp.join(self.data_dir, "camera", "*.json")))
        self.img_shape = self.get_img(0).shape[:2]
        self.ts = self.load_ts(self.cam_fs)
    
    def get_img(self, idx):
        return cv2.imread(self.img_fs[idx])

    def get_intrnxs(self):
        return load_json_intr(self.cam_fs[0])
    
    def load_ts(self, cam_fs):
        ts = []
        for i, cam_f in enumerate(cam_fs):
            with open(cam_f, "r") as f:
                data = json.load(f)
                try:
                    ts.append(data["t"])
                except Exception as e:
                    print(e)
                    print(osp.basename(cam_f), f"does not have t. frame: {i + 1}/{len(cam_fs)}. Replacing with last t")
                    ts.append(ts[-1])
        
        return np.array(ts)

    def get_extrnxs(self, idx):
        """
        returns world to cam
        """
        return load_json_cam(self.cam_fs[idx])
    
    def get_all_extrnxs(self):
        return np.stack([load_json_cam(e) for e in self.cam_fs[:self.__len__()]])

    def __len__(self):
        return min(len(self.cam_fs), len(self.img_fs))


class EcamSceneManager(ColcamSceneManager):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cam_fs = sorted(glob.glob(osp.join(self.data_dir, "camera", "*.json")))
        self.eimgs = np.load(osp.join(self.data_dir, "eimgs", "eimgs_1x.npy"), "r")

        prev_cam_dir = osp.join(self.data_dir, "prev_camera")
        if osp.exists(prev_cam_dir):
            self.prev_cam_fs = sorted(glob.glob(osp.join(prev_cam_dir, "*.json")))
            self.next_cam_fs = sorted(glob.glob(osp.join(self.data_dir, "next_camera", "*.json")))
            self.prev_ts = self.load_ts(self.prev_cam_fs)
            self.next_ts = self.load_ts(self.next_cam_fs)
            self.ts = self.prev_ts
        else:
            self.ts = self.load_ts(self.cam_fs)  # this one is slightly meaningless

    def __len__(self):
        return min(len(self.cam_fs), len(self.eimgs))

    def get_img(self, idx):
        img = np.stack([(self.eimgs[idx] != 0).astype(np.uint8) * 255]*3, axis=-1)
        return img


class EdEvimoManager(ColcamSceneManager):
    def __init__(self, data_dir):
        super().__init__(data_dir)

        depth_f = osp.join(data_dir, "depth.json")
        with open(depth_f, "r") as f:
            depths = json.load(f)
        ks = sorted(depths.keys())
        self.depths = [depths[k] for k in ks]
    
    def get_depth(self, idx):
        depth = self.depths[idx]
        return depth["min"], depth["max"]

    def get_all_depths(self):
        return np.array([[e["min"], e["max"]] for e in self.depths[:self.__len__()]])


class ColmapManager(SceneManager):
    def __init__(self, scene_dir):
        self.scene_dir = scene_dir
        self.img_dir = osp.join(scene_dir, "images")
        # self.img_dir = osp.join(scene_dir, "images")

        self.cam_f = osp.join(self.scene_dir, "sparse/0/cameras.bin")
        self._init_camera()

        self.colamp_img_f = osp.join(self.scene_dir, "sparse/0/images.bin")
        self.colmap_img = read_images_binary(self.colamp_img_f)
        self._init_extrinsics()

        img_names = [self.colmap_img[key].name for key in self.colmap_img.keys()]
        self.img_fs = sorted([osp.join(self.img_dir, img_name) for img_name in img_names])
        
    
    def _init_extrinsics(self):
        self.extrnsics = np.zeros((len(self.colmap_img), 3,4))

        for i, (k, v) in enumerate(self.colmap_img.items()):
            R, t = v.qvec2rotmat(), v.tvec[:, None]
            self.extrnsics[i] = np.concatenate([R, t], axis=-1)

    
    def _init_camera(self):
        cameras = read_cameras_binary(self.cam_f)
        camera = cameras[list(cameras.keys())[0]]
        
        fx, fy, cx, cy = camera.params[:4]
        self.D = camera.params[4:] if len(camera.params) > 4 else np.zeros(4)


        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        
        self.img_shape = (camera.height, camera.width)
    
    def __len__(self):
        return len(self.img_fs)

    def get_img_shape(self):
        return self.img_shape
    
    def get_img_f(self, idx):
        return self.img_fs[idx]
    
    def get_img(self, idx):
        return cv2.imread(self.img_fs[idx])
    
    def get_extrnxs(self, idx):
        return self.extrnsics[idx]
    
    def get_intrnxs(self):
        return self.K, self.D

MANAGER_DICT = {"colcam_set": ColcamSceneManager,
                "ecam_set": EcamSceneManager,
                "trig_ecamset": EcamSceneManager,
                "colmap": ColmapManager}

