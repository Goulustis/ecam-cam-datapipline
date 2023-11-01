import numpy as np
import cv2
import os
import os.path as osp
import glob
from tqdm import tqdm
from PIL import Image


def to_jpg(src_dir, targ_dir):
    os.makedirs(targ_dir, exist_ok=True)
    src_fs = sorted(glob.glob(osp.join(src_dir, "*")))

    for i, f in tqdm(enumerate(src_fs), total = len(src_fs)):
        targ_path = osp.join(targ_dir, str(i).zfill(6) + ".png")
        img = Image.open(f)
        img.save(targ_path)

def main():
    scale = 2 
    inp_fold = "rgb_checker"
    out_fold = f"{inp_fold}_{scale}"
    os.makedirs(out_fold, exist_ok=True)
    img_fs = sorted(glob.glob(osp.join(inp_fold, "*")))
    f_ext = img_fs[0].split(".")[-1]
    
    for i, img_f in tqdm(enumerate(img_fs), total=len(img_fs)):
        img = cv2.imread(img_f)
        h, w, c = img.shape
        n_h, n_w = h//scale, w//scale
        resized = cv2.resize(img, (n_w, n_h), cv2.INTER_AREA)

        # img_name = img_f.split("/")[-1]
        img_name = f'{str(i).zfill(6)}.{f_ext}'
        save_f = osp.join(out_fold, img_name)
        cv2.imwrite(save_f, resized)

if __name__ == "__main__":
    main()

