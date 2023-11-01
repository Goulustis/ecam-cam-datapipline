import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import pprint


def fill_triggers(fixed_trigs, ret_ones_only=False):
    """
    fill in the holes in the triggers, triggers could have 3 centers: [exposed, shutter_closed, exposed + shutter_closed]
    exposed + shutter_closed only have rising signal, missing the closing signal. This function will add in the time stamp for the 0 signal

    input:
        fixed_trigs (np.array [custom_type]): trigger that is already processed.
    
    return:
        fixed_ts (np.array): list of times with new timestamps injected ready for e2calib to use
        [not returned] fixed_ps (np.array): list of ps that is fixed, done for sanity check
    """
    t_diffs = np.diff(fixed_trigs['t'])
    uniq, cnt = np.unique(t_diffs, return_counts=True)
    uniq_idxs = np.arange(len(uniq))

    uniq_diffs = np.diff(uniq)

    uniq_cond = uniq_diffs > 4
    if uniq_cond.sum() == 1:
        # no need to fix since it only has [exposed, shutter_closed]
        return fixed_trigs['t'], fixed_trigs['p']
    
    exposed_t = uniq[:-1][uniq_cond][0]
    closed_t = uniq[:-1][uniq_cond][1] + 10

    # store the corrected ts
    fixed_ts = []
    fixed_ps = []
    
    trig_ts = fixed_trigs['t']
    trig_ps = fixed_trigs['p']
    st_t = trig_ts[0]
    st_p = trig_ps[0]

    for i in range(1, len(trig_ts)):
        end_t = trig_ts[i]
        end_p = trig_ps[i]
        t_diff = end_t - st_t 
        fixed_ts.append(st_t)
        fixed_ps.append(st_p)

        if t_diff > closed_t:
            new_t = st_t + exposed_t
            fixed_ts.append(new_t)
            fixed_ps.append(0)
            
        st_t = end_t 
        st_p = end_p
    fixed_ts.append(end_t)
    fixed_ps.append(end_p)

    # check if last time stamp is a 1 or 0; if 1, it missed the falling edge;
    # add the extra time stamp back in
    if fixed_trigs['p'][-1] == 1:
        fixed_ts.append(end_t + exposed_t)
        fixed_ps.append(0)

    fixed_ts = np.array(fixed_ts)
    fixed_ps = np.array(fixed_ps)

    print("injected ts diffs:")
    uniq, cnt = np.unique(np.diff(fixed_ts), return_counts=True)
    pprint.pprint(dict(zip(uniq, cnt)))
    print("\n")

    # the number of triggers could be no even

    if ret_ones_only:
        return fixed_ts[fixed_ps == 1]
    else:
        return fixed_ts, fixed_ps
            

def save_alt_trigger_options(inj_ts:np.ndarray, inj_ps:np.ndarray, trig_fold:str):
    st_ts = inj_ts[inj_ps == 1]
    end_ts = inj_ts[inj_ps == 0]
    mean_ts = ((st_ts + end_ts)/2).astype(int)

    np.savetxt(osp.join(trig_fold, "st_triggers.txt"), st_ts)
    np.savetxt(osp.join(trig_fold, "end_triggers.txt"), end_ts)
    np.savetxt(osp.join(trig_fold, "mean_triggers.txt"), mean_ts)
    

def main():
    """
    remove recorded duplicate triggers
    inject missing triggers in between
    save only "1" tirggers only
    """
    viz = False

    # trig_fold = "raw_events/rgb_checker" 
    trig_fold = "raw_events/dev_trig"
    trigger_f = osp.join(trig_fold, "triggers.npy")
    ori_trigger = np.load(trigger_f)

    cond = (np.diff(ori_trigger["t"]) < 200)
    print("num zeros:", cond.sum())

    trigger = np.concatenate([ori_trigger[0:1], ori_trigger[1:][~cond]])

    np.save(osp.join(trig_fold, "triggers_fixed.npy"), trigger)
    np.savetxt(osp.join(trig_fold, "triggers.txt"), fill_triggers(trigger, ret_ones_only=True))

    t_diffs = np.diff(trigger["t"])
    uniq, cnt = np.unique(t_diffs, return_counts=True)

    print("if the below is centered around 2 or 3 mean, the triggers are fixed")
    pprint.pprint(dict(zip(uniq, cnt)))
    print("\n")

    injected_ts, injected_ps = fill_triggers(trigger)
    save_alt_trigger_options(injected_ts, injected_ps, trig_fold)

    if viz:
        fig, axs = plt.subplots(1,2)
        to_col = lambda x : np.concatenate([x[...,None], np.zeros((len(x),2))], axis=-1)
        axs[0].scatter(np.arange(len(ori_trigger)), ori_trigger["t"], c=to_col(ori_trigger["p"]))
        axs[1].scatter(np.arange(len(trigger)), trigger["t"], c=to_col(trigger["p"]))
        plt.show()

if __name__ == "__main__":
    main()
