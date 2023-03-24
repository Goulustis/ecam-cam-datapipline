import numpy as np
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

    pos_p = p==1
    neg_p = p==0

    e_img = np.zeros((h,w), dtype=np.int32)
    
    np.add.at(e_img, (y[pos_p], x[pos_p]), 1)
    np.add.at(e_img, (y[neg_p], x[neg_p]), -1)
    
    assert np.abs(e_img).max() < np.iinfo(np.int8).max, "type needs to be bigger"

    return e_img

def synthesize_fake_triggers(evs_end_t, trig_st=0, n_eimg_per_gap=4, time_delta=5000):
    """
    When no triggers is provided, synthesize a set of fake triggers that splits the entire
    event stream 

    input:
        evs_end_t (int): last time stamp of event
        trig_st (int):   start time of synthetic trigger
        n_eimg_per_gap (int) : number of event image per gap
        time_delta(int)  : the time window to accumulate events by
    """
    n_eimg_per_gap = 4
    trig_delta = n_eimg_per_gap*time_delta
    trig_st = 0
    triggers = [trig_st+i*trig_delta for i in range((evs_end_t - trig_st)//trig_delta)]
    return triggers


def create_event_imgs(events, triggers=None, time_delta=5000, create_imgs = True, st_t=0):
    """
    input:
        events (np.array [type (t, x, y, p)]): events
        triggers (np.array [int]): list of trigger time; will generate tight gap if none
        time_delta (int): time in ms, the time gap to create event images
        st_t (int): starting time to accumulate the event images
        create_imgs (bool): actually create the event images, might use this function just to
                            get timestamps and ids

    return:
        eimgs (np.array): list of images with time delta of 50
        eimgs_ts (np.array): list of time at which the event image is accumulated to
        eimgs_ids (np.array): list of embedding ids for each image
        trigger_ids (np.array): list of embedding ids of each trigger time
    """
    if create_imgs:
        print("creating event images")
    else:
        print("not creating event images, interpolating cameras and creating ids only")

    # if (events is not None):
    #     t = events["t"]
    #     cond = t >= st_t
    #     t, x, y, p = events["t"][cond], events["x"][cond], events["y"][cond], events["p"][cond]
    
    
    if triggers is not None:
        # number of eimg per trigger gap
        n_eimg_per_gap = int((triggers[1] - triggers[0]) // time_delta)
    else:
        n_eimg_per_gap = 4
        triggers = synthesize_fake_triggers(t[-1], n_eimg_per_gap=n_eimg_per_gap, 
                                            trig_st=st_t)

    eimgs = []       # the event images
    eimgs_ts = []    # timestamp of event images
    eimgs_ids = []   # embedding ids for each img
    trig_ids = []    # id at each trigger

    id_cnt = 0
    with tqdm(total=(len(triggers) - 1)) as pbar:
        for trig_idx in range(1, len(triggers)):
            trig_st, trig_end = triggers[trig_idx - 1], triggers[trig_idx]

            if (events is not None) and create_imgs:
                curr_t, curr_x, curr_y, curr_p = events.retrieve_data(trig_st, trig_end)

            

            st_t = trig_st
            end_t = trig_st + time_delta
            trig_ids.append(id_cnt)

            while st_t < trig_end:
                if (events is not None) and create_imgs:
                    cond = (st_t <= curr_t) & (curr_t <= end_t)
                    eimg = ev_to_img(curr_x[cond], curr_y[cond], curr_p[cond])
                    eimgs.append(eimg)

                eimgs_ids.append(id_cnt)
                # eimgs_ts.append(st_t)
                eimgs_ts.append(int((st_t + end_t)/2))

                # update
                st_t = end_t
                end_t = end_t + time_delta
                end_t = min(end_t, trig_end)
                id_cnt += 1

            pbar.update(1)

    if (events is not None) and create_imgs:
        return np.stack(eimgs), np.array(eimgs_ts, dtype=np.int32), np.array(eimgs_ids, dtype=np.int32), np.array(trig_ids, dtype=np.int32)
    else:
        return None, np.array(eimgs_ts), np.array(eimgs_ids, dtype=np.int32), np.array(trig_ids, dtype=np.int32)




# PLAN:
# 1) use img_times as points for interpolation
# 2) use trigger ids for warp embd id in color dataset