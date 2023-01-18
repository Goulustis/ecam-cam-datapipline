import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def ev_to_img(x, y, p, e_thresh=0.15):
    """
    input:
        evs (np.array [type (t, x, y, p)]): events such that t in [t_st, t_st + time_delta]
    return:
        event_img (np.array): of shape (h, w)
    """
    e_thresh = 0.15
    h, w = 720, 1280
    # x, y, p = evs["x"], evs["y"], evs["p"]

    pos_p = p==1
    neg_p = p==0

    # e_img = np.zeros((h,w), dtype=np.int32)
    e_img = np.zeros((h,w), dtype=np.float16)
    cnt = np.zeros((h,w), dtype=np.int32)

    e_img[y[pos_p], x[pos_p]] += e_thresh
    e_img[y[neg_p], x[neg_p]] -= e_thresh
    cnt[y,x] += 1

    stat = (e_img.max(), e_img.min())
    # assert e_img.max() <= 127, "event image value overflow"
    # assert e_img.min() >= -128, "event image value underflow"

    # e_img = e_img.astype(np.int8)
    return e_img

def create_event_imgs(events, triggers, time_delta=5000, create_imgs = True):
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
    n_eimg_per_gap = int((triggers[1] - triggers[0]) // time_delta)
    eimgs = []       # the event images
    eimgs_ts = []
    eimgs_ids = []   # embedding ids for each img
    trig_ids = []    # id at each trigger

    if events is not None:
        t = events["t"]
        x, y, p = events["x"], events["y"], events["p"]

    id_cnt = 0
    with tqdm(total=n_eimg_per_gap*len(triggers)) as pbar:
        # for trig_t in triggers:
        for trig_idx in range(1, len(triggers)):
            trig_st, trig_end = triggers[trig_idx - 1], triggers[trig_idx]

            if events is not None:
                cond = (trig_st <= t) & (t <= trig_end)
                curr_t, curr_x, curr_y, curr_p = [e[cond] for e in [t, x, y, p]]
            

            st_t = trig_st
            end_t = trig_st + time_delta
            trig_ids.append(id_cnt)

            for _ in range(n_eimg_per_gap):
                if (events is not None) and create_imgs:
                    cond = (st_t <= curr_t) & (curr_t <= end_t)
                    # img_events = events[cond]
                    # eimg = ev_to_img(img_events)
                    eimg = ev_to_img(curr_x[cond], curr_y[cond], curr_p[cond])
                    eimgs.append(eimg)


                eimgs_ids.append(id_cnt)
                eimgs_ts.append(st_t)

                # update
                st_t = end_t
                end_t = end_t + time_delta
                id_cnt += 1

                pbar.update(1)

    if (events is not None) and create_imgs:
        return np.stack(eimgs), np.array(eimgs_ts, dtype=np.int32), np.array(eimgs_ids, dtype=np.int32), np.array(trig_ids, dtype=np.int32)
    else:
        return None, np.array(eimgs_ts, dtype=np.int32), np.array(eimgs_ids, dtype=np.int32), np.array(trig_ids, dtype=np.int32)




# PLAN:
# 1) use img_times as points for interpolation
# 2) use trigger ids for warp embd id in color dataset