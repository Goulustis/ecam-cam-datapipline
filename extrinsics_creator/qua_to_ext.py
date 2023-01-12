import sys
sys.path.append(".")

from colmap_find_scale.read_write_model import read_images_binary
import numpy as np
import json
from tqdm import tqdm
import os.path as osp

import argparse

def main(imgs_f, save_path=None):
    dirname = osp.dirname(imgs_f)
    imgs = read_images_binary(imgs_f)

    keys = sorted(list(imgs.keys()))

    ext_mtxs = []
    for k in tqdm(keys):
        img = imgs[k]
        R = img.qvec2rotmat() 
        t = img.tvec[...,None]
        
        mtx = np.concatenate([R,t], axis=-1)
        dummy = np.zeros((1,4))
        dummy[0,-1] = 1
        mtx = np.concatenate([mtx, dummy], axis=0)
        ext_mtxs.append(mtx)

    ext_mtxs = np.stack(ext_mtxs)

    if save_path is None:
        np.save(osp.join(dirname, "images_mtx.npy"), ext_mtxs)
    else:
        np.save(save_path, ext_mtxs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="turn colmap quaterions to extrinsic matrixes")
    parser.add_argument("-i", "--img_path", help="path to colmap images.bin")
    parser.add_argument("-o", "--output_path", help="path to save extrinsic cameras", default=None)
    args = parser.parse_args()

    main(args.img_path, args.output_path)
