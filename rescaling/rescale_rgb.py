import cv2
from tqdm import tqdm
import glob
import os.path as osp
import os 
import argparse
from tqdm import tqdm

def main(img_dir, scale = 2):
    """
    assume img_dir contains original size images
    
    inputs:
        img_dir (str): path to image dir
        scale (int): scale to reduce image size by
    """
    targ_dir = osp.join(osp.dirname(img_dir), f'{scale}x')
    os.makedirs(targ_dir, exist_ok=True)    

    img_fs = sorted(glob.glob(osp.join(img_dir, "*")))
    h, w = cv2.imread(img_fs[0]).shape[:2]
    new_h, new_w = h//2, w//2

    for f in tqdm(img_fs, desc="resizing rgb"):
        img = cv2.imread(f)
        re_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        save_path = osp.join(targ_dir, osp.basename(f))
        cv2.imwrite(save_path, re_img)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="data/formatted_checkers/colcam_set/rgb/1x")
    parser.add_argument("--scale", type=float, default=2)
    args = parser.parse_args()

    main(args.img_dir, args.scale)
