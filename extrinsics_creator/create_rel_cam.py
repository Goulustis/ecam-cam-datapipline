import json
import numpy as np
from tqdm import tqdm
import argparse
import os.path as osp

def read_rel_cam():
    with open("cam_rel.json", "r") as f:
        data = json.load(f)
    
    for k, v in data.items():
        data[k] = np.array(v)
    
    return data

def read_col_cam():
    with open("col_recon/extrinsic_mtx.json", "r") as f:
        data = json.load(f)
    
    keys = list(data.keys())
    keys = sorted(keys)
    cams = []
    for i in keys:
        v = data[str(i)]
        cams.append(np.array(v["ext_mtx"]))
    
    return np.stack(cams)

def map_cam(rel_cam, col_cams, scale):
    R, T = rel_cam["R"], rel_cam["T"]
    T = T*scale
    new_cams = []

    # pointless to optimize any faster
    for i, col_cam in tqdm(enumerate(col_cams), desc="mapping cams", total=len(col_cams)):
        R1 = col_cam[:3, :3]
        T1 = col_cam[:3, -1][..., None]
        
        # https://stackoverflow.com/questions/38737960/finding-relative-rotation-between-two-cameras
        R2 = R@R1
        T2 = R@T1 + T

        new_cam = np.concatenate([R2, T2], axis = -1)
        u = np.zeros(4)
        u[-1] = 1
        new_cam = np.concatenate([new_cam, u[None]])

        new_cams.append(new_cam)

    return np.stack(new_cams)
    

def main():
    parser = argparse.ArgumentParser(description="create the event camera extrinsics")
    parser.add_argument("-r", "--rel_cam_path", help="path to rel_cam.json")
    parser.add_argument("-c", "--col_cam_path", help="path to images_mtx.npy, the extrinsic matrix of colmap")
    parser.add_argument("-s", "--scale_path", help="path to colmap_scale.txt containing the scale of colmap scene in (unit/{mm, cm, m, ...})")
    args = parser.parse_args()

    rel_cam = read_rel_cam(args.r)
    col_cams = np.load(args.c)
    with open(args.s, "r") as f:
        scale = float(f.readline())

    e_cams = map_cam(rel_cam, col_cams, scale)
    f = lambda x : osp.dirname(x)
    save_path = osp.join(f(f(f(args.c))), "e_cams.npy")
    np.save(save_path, e_cams)

if __name__ == "__main__":
    main()