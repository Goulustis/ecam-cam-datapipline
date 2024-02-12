import sys
sys.path.append(".")

import numpy as np
import json
import h5py
import os.path as osp
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import bisect


class EventBuffer:
    def __init__(self, ev_f) -> None:
        self.ev_f = ev_f
        self.x_f, self.y_f, self.p_f, self.t_f = self.load_events(self.ev_f)

        self.fs = [self.x_f, self.y_f, self.p_f, self.t_f]

        self.n_retrieve = 5000000
        self._init_cache(0)


    def _init_cache(self, idx=0):
        self.x_cache = np.array([self.x_f[idx]])
        self.y_cache = np.array([self.y_f[idx]])
        self.t_cache = np.array([self.t_f[idx]])
        self.p_cache = np.array([self.p_f[idx]])

        self.caches = [self.x_cache, self.y_cache, self.t_cache, self.p_cache]

        self.curr_pnter = idx + 1
    
    def clear_cache(self):
        self.x_cache = np.array([])
        self.y_cache = np.array([])
        self.t_cache = np.array([])
        self.p_cache = np.array([])
    
    def load_events(self, ev_f):
        self.f = h5py.File(ev_f, "r")
        x_f = self.f["x"]
        y_f = self.f["y"]
        p_f = self.f["p"]
        t_f = self.f["t"]

        return x_f, y_f, p_f, t_f

    
    def update_cache(self):
        
        rx, ry, rp, rt = [e[self.curr_pnter:self.curr_pnter + self.n_retrieve] for e in self.fs]
        self.x_cache = np.concatenate([self.x_cache, rx])
        self.y_cache = np.concatenate([self.y_cache, ry])
        self.p_cache = np.concatenate([self.p_cache, rp])
        self.t_cache = np.concatenate([self.t_cache, rt])
        
        self.curr_pnter = min(len(self.t_f), self.curr_pnter + self.n_retrieve)

    def drop_cache_by_cond(self, cond):
        self.x_cache = self.x_cache[cond]
        self.y_cache = self.y_cache[cond]
        self.p_cache = self.p_cache[cond]
        self.t_cache = self.t_cache[cond]

    def retrieve_data(self, st_t, end_t, is_far=False):
        if (self.t_cache[0] > st_t) or is_far:
            ## if st_t already out of range
            idx = bisect.bisect(self.t_f, st_t)
            idx = idx if ((st_t == self.t_f[idx]) or st_t <= self.t_f[0]) else idx - 1

            assert idx >= 0, f"{st_t} not found!!"

            self._init_cache(idx)


        while (self.curr_pnter < len(self.t_f)) and (self.t_cache[-1] <= end_t):
            self.update_cache()
        
        ret_cond = ( st_t<= self.t_cache) & (self.t_cache <= end_t)
        ret_data = [self.t_cache[ret_cond], self.x_cache[ret_cond], 
                    self.y_cache[ret_cond], self.p_cache[ret_cond]]
        self.drop_cache_by_cond(~ret_cond)

        return ret_data
    
    def drop_cache_by_t(self, t):
        cond = self.t_cache >= t
        self.drop_cache_by_cond(cond)

    # def pass_end(self):
    #     return self.curr_pnter >= len(self.t_f)
    


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
        cam_i (int): one of [1, 2], 1 for color camera, 2 for event camera
    output:
        M (np.array): 3x3 intrinsic matrix
        dist (list like): distortion (k1, k2, p1, p2, k3)
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    dist = data[f"dist{cam_i}"]
    dist = dist if type(dist[0]) != list else dist[0]
    return np.array(data[f"M{cam_i}"]), dist

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
        np_path = osp.join(osp.dirname(path), "events.npy")
        if osp.exists(np_path):
            return np.load(np_path)

        with h5py.File(path, "r") as f:
            xs,ys,ts,ps = [f[e][:] for e in list("xytp")]
        
        return {"x": xs, "y":ys, "t":ts, "p":ps}

    else:
        raise Exception("event file format not supported")
        
        
