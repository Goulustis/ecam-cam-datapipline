import numpy as np
import cv2
import os.path as osp
import os 
from tqdm import tqdm 
import argparse

"""
assume eimgs is at 1x scale
"""

def main(eimg_path, scale):
    print("loading eimgs")
    eimgs = np.load(eimg_path)
    print("dtype", eimgs.dtype)

    targ_path = osp.join(osp.dirname(eimg_path), f"eimgs_{scale}x")

    n, h, w = eimgs.shape[:3]
    new_h, new_w = h//scale, w//scale

    re_eimgs = np.zeros((n, new_h, new_w), dtype=eimgs.dtype)

    for i, eimg in tqdm(enumerate(eimgs), total=len(eimgs), desc="resize eimgs"):
        eimg = eimg.astype(np.float32)
        re_img = cv2.resize(eimg,(new_w, new_h), interpolation=cv2.INTER_AREA)
        eimg = eimg.astype(np.float16)

        re_eimgs[i] = re_img
    
    np.save(targ_path, re_eimgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eimg_path", default="data/formatted_checkers/ecam_set/eimgs/eimgs_1x.npy")
    parser.add_argument("--scale", type=float, default=2)
    args = parser.parse_args()

    main(args.eimg_path, args.scale)
