import sys
sys.path.append(".")

from data_utils.process_event_format import EventCD
import numpy as np
import json
import h5py
import os.path as osp
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool


def read_triggers(path):

    if path is None:
        return None
            
    trigs = []
    with open(path, "r") as f:
        for l in f:
            trigs.append(float(l.rstrip("\n")))

    return np.array(trigs)


def read_ecam_intrinsics(path, cam_i = 2):
    """
    input:
        path (str): path to json
    output:
        M (np.array): 3x3 intrinsic matrix
        dist (list like): distortion (k1, k2, p1, p2, k3)
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    return np.array(data[f"M{cam_i}"]), data[f"dist{cam_i}"][0]

def read_events(path, save_np = False, targ_dir = None):
    """
    input:
        path (str): path to either a h5 or npy 
        make_np (bool): if path is h5, make a numpy copy of it after reading 
    return:
        data (np.array [EventCD]): return events of type EventCD
    """
    if ".npy" in path:
        return np.load(path)

    elif ".h5" in path:
        np_path = osp.join(osp.basename(path), "events.npy")
        if osp.exists(np_path):
            return np.load(np_path)

        with h5py.File(path, "r") as f:
            xs,ys,ts,ps = [f[e][:] for e in list("xytp")]
        
        return {"x": xs, "y":ys, "t":ts, "p":ps}

    else:
        raise Exception("event file format not supported")
        
        
