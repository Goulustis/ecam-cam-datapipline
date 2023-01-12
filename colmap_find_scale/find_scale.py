import numpy as np
from read_write_model import read_points3D_binary
import argparse
import os.path as osp

def read_obs_pnt(obs_pnt_f):
    """
    input:
        obs_pnt_f (str): path to chosen points
    return:
        pnts (list[float]): idx of colmap 3dpoints
    """

    with open(obs_pnt_f, "r") as f:
        pnts = [int(e) for e in f.readlines()]
    
    return pnts

def pnt_3d_split(pnts_3d):
    """
    input:
        pnts_3d (dict:int --> colmap_3dpoints)

    return:
        idxs (np.array): idx of all 3d points
        locs (np.array): (x,y,z) of all points
    """
    idxs = []
    locs = []
    for k, p in pnts_3d.items():
        idxs.append(k)
        locs.append(p.xyz)
    
    return np.array(idxs), np.stack(locs)


def get_neighbors(locs, pnt, eps=0.07):
    """
    find neighbor points in locs
    """
    cond = np.sqrt(((locs-pnt[None])**2).sum(axis=-1)) <= eps
    return locs[cond] 

def read_r_len(r_len_f):
    """
    input:
        r_len_f (str): path to real length txt
    """

    l = -1
    with open(r_len_f, "r") as f:
        l = float(f.readline())

    return l

def find_scale(col_pnt_f, obs_pnt_f, r_len_f):
    """
    input:
        col_pnt_f (str): path to point3D.bin
        obs_pnt_f (str): path to points chosen for finding the scale
        r_len (float): actual distance bewteen the chosen points {unit}
    output:
        scale (float): the scale of the scene: col_unit/{unit}
    """
    pnts_3d = read_points3D_binary(col_pnt_f)
    _, locs = pnt_3d_split(pnts_3d)
    
    c_idx = read_obs_pnt(obs_pnt_f)
    r_len = read_r_len(r_len_f)

    a_pnts = get_neighbors(locs, pnts_3d[c_idx[0]].xyz)
    b_pnts = get_neighbors(locs, pnts_3d[c_idx[1]].xyz)

    n_use = min(len(a_pnts), len(b_pnts))
    a_pnts = a_pnts[:n_use]
    b_pnts = b_pnts[:n_use]

    return np.sqrt(((a_pnts - b_pnts)**2).sum(axis=1)).mean()/r_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="finds the scale of colmap scene by sampling surrounding chosen points")
    parser.add_argument("-c", "--col_pts_path", help="path to point3D.bin file from colmap",default="data/checker/checker_recon/sparse/0/points3D.bin")
    parser.add_argument("-p", "--pnt_txt_path", help="txt containing the id of the 3d points in colmap",default="data/checker/checker_recon/scale_pnts.txt")
    parser.add_argument("-l", "--len_path", help="a .txt file containing the actual length of the chosen scalingn point", default="data/checker/checker_recon/pnt_dist.txt")
    args = parser.parse_args()

    scale = find_scale(args.col_pts_path, args.pnt_txt_path, args.len_path)

    f_d = lambda x : osp.dirname(x)
    scale_path = osp.join(osp.dirname(args.pnt_txt_path), "colmap_scale.txt")
    
    with open(scale_path, "w") as f:
        f.write(str(scale))