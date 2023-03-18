import sys
sys.path.append(".")

import numpy as np
from tqdm import tqdm
from format_data.utils import read_events, read_triggers
import os.path as osp


linked_Type = np.dtype({'names':['x','y','p','t', "t_prev", "next_idx"], 
                        'formats':['<u2','<u2','i2', 'u4', 'u4', 'i4']})

def main():
    """
    turn all events into tuple of: (x, y, p, t, t_prev, next_idx),
                                   next_idx = -1 if event is last event
    """
    ev_path = "data/mean_t_rgb_checker/ecam_set/processed_events.h5"
    trigger_path = "data/rgb_checker/triggers.txt"

    trigger_st = read_triggers(trigger_path)[0]
    evs = read_events(ev_path)
    xs, ys ,ts, ps = [evs[e] for e in list("xytp")]
    h, w = ys.max() + 1, xs.max() + 1
    t_min = ts.min()
    assert t_min == ts[0], "t is unordered! something is wrong"

    keep_cond = ts > trigger_st
    xs, ys, ts, ps = [e[keep_cond] for e in [xs, ys, ts, ps]]

    prev_ts = np.full((h, w), trigger_st)
    prev_idxs = np.full((h, w), -1)
    # lined_data = []
    lined_data = np.empty(len(xs), dtype=linked_Type)
    for curr_idx, (x,y,t,p) in tqdm(enumerate(zip(xs,ys,ts,ps)), total=len(xs), desc="lining data"):
        prev_t = prev_ts[y, x]
        curr_ev = np.array([(x, y, p, t, prev_t, -1)], dtype=linked_Type)

        prev_idx = prev_idxs[y, x]
        if prev_idx != -1:
            prev_ev = lined_data[prev_idx]
            prev_ev[-1] = curr_idx
        prev_idxs[y, x] = curr_idx

        lined_data[curr_idx] = curr_ev
    
    lined_data = np.array(lined_data, dtype=linked_Type)
    save_path = osp.join(osp.dirname(ev_path), "linked_events.npy")
    np.save(save_path, lined_data)


if __name__ == "__main__":
    main()
