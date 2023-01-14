import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def ev_to_img(evs, eps):
    """
    input:
        evs (np.array [type (t, x, y, p)]): events such that t in [t_st, t_st + time_delta]
    return:
        event_img (np.array): of shape (h, w)
    """
    e_thresh = 0.15
    h, w = 720, 1280
    x, y, p = evs["x"], evs["y"], evs["p"]

    pos_p = p==1
    neg_p = p==0

    e_img = np.zeros((h,w))

    e_img[y[pos_p], x[pos_p]] += e_thresh
    e_img[y[neg_p], x[neg_p]] -= e_thresh

    e_img = e_img.reshape(h,w)

    img = np.zeros((h,w,3), dtype=np.uint8)
    img[e_img>0, 1] = 255
    img[e_img<0, 2] = 255
    img[e_img == 0] = 255

    return img

def create_event_imgs(events, triggers, time_delta=50):
    """
    input:
        events (np.array [type (t, x, y, p)]): events
        time_delta (int): time in ms, the time gap to create event images
        st_t (int): starting time to accumulate the event images
        triggers (np.array [int]): list of trigger time

    return:
        event_imgs (np.array): list of images with time delta of 50
        img_times (np.array): list of time at which the event image is accumulated to
        img_ids (np.array): list of embedding ids for each image
        trigger_ids (np.array): list of embedding ids of each trigger time
    """

    # number of eimg per trigger gap
    n_eimg_per_gap = (triggers[1] - triggers[0]) // time_delta
    eimgs = []
    eimgs_ids = []
    trig_ids = []

    id_cnt = 0
    for trig_t in triggers:
        st_t = trig_t
        end_t = trig_t + time_delta
        trig_ids.append(id_cnt)

        for _ in range(n_eimg_per_gap):
            cond = 3
    
    pass



# PLAN:
# 1) use img_times as points for interpolation
# 2) use trigger ids for warp embd id in color dataset