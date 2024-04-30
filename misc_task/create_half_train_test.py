import glob
import os
import os.path as osp
import cv2
import json
import copy
from tqdm import tqdm

class HalfFormatter:
    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.dst_dir = osp.join(osp.dirname(src_dir), "half_colcam_set")
        self.dst_rgb_dir = osp.join(self.dst_dir, "rgb/1x")
        self.dst_cam_dir = osp.join(self.dst_dir, "camera")

        self.img_fs = sorted(glob.glob(osp.join(src_dir, "rgb/1x/*.png")))
        self.cam_fs = sorted(glob.glob(osp.join(src_dir, "camera/*.json")))

        self.n_img_digits = len(osp.basename(self.img_fs[0]).split(".")[0])
        self.n_cam_digits = len(osp.basename(self.cam_fs[0]).split(".")[0])
        self.ori_img_shape = cv2.imread(self.img_fs[0]).shape[:2] # (h, w)

        os.makedirs(self.dst_rgb_dir, exist_ok=True), os.makedirs(self.dst_cam_dir, exist_ok=True)
        
        with open(osp.join(src_dir, "dataset.json")) as f:
            self.dataset = json.load(f)
        
        self.dst_dataset = copy.deepcopy(self.dataset)
        del self.dst_dataset["val_ids"]
        
        self.train_ids = [int(e) for e in self.dataset["train_ids"]]
        self.val_ids = [int(e) for e in self.dataset["val_ids"]]
        self.test_ids = [int(e) for e in self.dataset["test_ids"]]
        self.n_half_created = 0
    
    def create_train_val_half(self):
        print("halving images")
        self.half_and_save_imgs()

        print("modifying cameras")
        self.half_and_save_cameras()

        print("saving dataset json")
        self.save_dataset()

        print("modifying metadata")
        self.modify_and_save_metadata()

        print("making dummy dataset")
        self.create_dummy_data()


    def modify_and_save_metadata(self):
        src_metadata_f = osp.join(self.src_dir, "metadata.json")
        dst_metadata_f = osp.join(self.dst_dir, "metadata.json")

        with open(src_metadata_f) as f:
            metadata = json.load(f)
        
        dst_metadata = copy.deepcopy(metadata)
        val_meta = [metadata[e] for e in self.dataset["val_ids"]]
        
        for i, top_half_idx in enumerate(self.dst_dataset["half_train_ids"]):
            dst_metadata[top_half_idx] = val_meta[i]
        
        for i, bot_half_idx in enumerate(self.dst_dataset["val_ids"]):
            dst_metadata[bot_half_idx] = val_meta[i]

        with open(dst_metadata_f, "w") as f:
            json.dump(dst_metadata, f, indent=2)
        

    
    def save_dataset(self):
        with open(osp.join(self.dst_dir, "dataset.json"), "w") as f:
            json.dump(self.dst_dataset, f, indent=2)

    def create_dummy_data(self):
        """
        because ednerf uses num_train to initialize num embeddings, need to make dummy data to be consistent during training
        """
        dummy_cam_fs = self.cam_fs[self.n_half_created:]
        dummy_img_fs = self.img_fs[self.n_half_created:]

        for cam_f, img_f in tqdm(zip(dummy_cam_fs, dummy_img_fs), total=len(dummy_cam_fs)):
            cam_f, img_f = osp.abspath(cam_f), osp.abspath(img_f)
            dst_cam_f = osp.abspath(osp.join(self.dst_cam_dir, osp.basename(cam_f)))
            dst_img_f = osp.abspath(osp.join(self.dst_rgb_dir, osp.basename(img_f)))

            os.symlink(cam_f, dst_cam_f), os.symlink(img_f, dst_img_f)

            

    def half_and_save_cameras(self):
        val_cam_fs = [self.cam_fs[i] for i in self.val_ids]
        h, w = self.ori_img_shape
        
        train_cameras = []
        val_cameras = []
        for cam_f in val_cam_fs:
            with open(cam_f) as f:
                cam = json.load(f)
                cam["image_size"] = [w, h//2]
            train_cameras.append(copy.deepcopy(cam)), val_cameras.append(copy.deepcopy(cam))
        
        cam_idx = 0
        for train_cam in train_cameras:
            save_f = osp.join(self.dst_cam_dir, f"{str(cam_idx).zfill(self.n_cam_digits)}.json")
            with open(save_f, "w") as f:
                json.dump(train_cam, f, indent=2)
            cam_idx += 1
        
        val_ids = []
        for val_cam in val_cameras:
            val_ids.append(str(cam_idx).zfill(self.n_cam_digits))
            save_f = osp.join(self.dst_cam_dir, f"{str(cam_idx).zfill(self.n_cam_digits)}.json")
            val_cam["principal_point"] = [val_cam["principal_point"][0], val_cam["principal_point"][1] - h//2]
            with open(save_f, "w") as f:
                json.dump(val_cam, f, indent=2)
            cam_idx += 1


    def half_and_save_imgs(self):
        val_img_fs = [self.img_fs[i] for i in self.val_ids]
        
        train_imgs = []
        val_imgs = []
        for img_f in val_img_fs:
            img = cv2.imread(img_f)
            h, w = img.shape[:2]
            train_imgs.append(img[:h//2])  # train is top half
            val_imgs.append(img[h//2:])    # val is bottom half
        
        img_idx = 0
        half_train_ids = []
        for train_img in train_imgs:
            half_train_ids.append(str(img_idx).zfill(self.n_img_digits))
            save_f = osp.join(self.dst_rgb_dir, f"{str(img_idx).zfill(self.n_img_digits)}.png")
            cv2.imwrite(save_f, train_img)
            img_idx += 1
        
        val_ids = []
        for val_img in val_imgs:
            val_ids.append(str(img_idx).zfill(self.n_img_digits))
            save_f = osp.join(self.dst_rgb_dir, f"{str(img_idx).zfill(self.n_img_digits)}.png")
            cv2.imwrite(save_f, val_img)
            img_idx += 1

        self.dst_dataset["val_ids"] = val_ids
        self.dst_dataset["half_train_ids"] = half_train_ids
        self.n_half_created = len(half_train_ids) + len(val_ids)


def main(src_dir):
    """
    src_dir (str): assume <scene>/colcam_set
    """
    formatter = HalfFormatter(src_dir)
    formatter.create_train_val_half()

if __name__ == "__main__":
    src_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/depth_var_1_lr_000000/colcam_set"
    main(src_dir)


