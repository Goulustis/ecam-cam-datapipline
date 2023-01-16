import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def ev_to_img(evs, e_thresh=0.15):
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


    return e_img

def create_event_imgs(events, triggers, time_delta=50, create_imgs = True):
    """
    input:
        events (np.array [type (t, x, y, p)]): events
        time_delta (int): time in ms, the time gap to create event images
        st_t (int): starting time to accumulate the event images
        triggers (np.array [int]): list of trigger time
        create_imgs (bool): actually create the event images, might use this function just to
                            get timestamps and ids

    return:
        eimgs (np.array): list of images with time delta of 50
        eimgs_ts (np.array): list of time at which the event image is accumulated to
        eimgs_ids (np.array): list of embedding ids for each image
        trigger_ids (np.array): list of embedding ids of each trigger time
    """

    # number of eimg per trigger gap
    n_eimg_per_gap = (triggers[1] - triggers[0]) // time_delta
    eimgs = []       # the event images
    eimgs_ts = []
    eimgs_ids = []   # embedding ids for each img
    trig_ids = []    # id at each trigger

    if events is not None:
        ts = events["t"]

    id_cnt = 0
    for trig_t in triggers:
        st_t = trig_t
        end_t = trig_t + time_delta
        trig_ids.append(id_cnt)

        for _ in range(n_eimg_per_gap):
            if create_imgs:
                cond = (st_t <= ts) & (ts <= end_t)
                img_events = events[cond]
                eimg = ev_to_img(img_events)
                eimgs.append(eimg)


            eimgs_ids.append(id_cnt)
            eimgs_ts.append(st_t)

            # update
            st_t = end_t
            end_t = end_t + time_delta
            id_cnt += 1

    if create_imgs:
        return np.stack(eimgs), np.array(eimgs_ts), np.array(eimgs_ids), np.array(trig_ids)
    else:
        return None, np.array(eimgs_ts), np.array(eimgs_ids), np.array(trig_ids)




# PLAN:
# 1) use img_times as points for interpolation
# 2) use trigger ids for warp embd id in color dataset