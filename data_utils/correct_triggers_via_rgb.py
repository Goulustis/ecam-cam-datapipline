import numpy as np
import argparse
import os.path as osp
import glob
import shutil

def to_microsec(time_str):
    """
    Converts a time string in the format "hh_mm_ss_mmm" to microseconds.
    Here, hh = hours, mm = minutes, ss = seconds, mmm = milliseconds.
    """
    # Splitting the string into hours, minutes, seconds, and milliseconds
    hours_str, minutes_str, seconds_str, milliseconds_str = time_str.split("_")

    # Converting each part to integers
    hours = int(hours_str)
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    milliseconds = int(milliseconds_str)

    # Calculating total microseconds
    total_microseconds = (hours * 3600 + minutes * 60 + seconds) * 1000 * 1000 + milliseconds * 1000

    return total_microseconds


def parse_raw_ts(raw_fs):
    raw_ts = np.array([to_microsec(osp.basename(f).split(".")[0]) for f in raw_fs])
    return raw_ts


def fix_trigger(triggers:np.ndarray, raw_ts:np.ndarray, thresh:float=4500.0):
    raw_ts = raw_ts - raw_ts[0] + triggers[0]
    trig_idx, raw_idx = 0, 0

    new_trigs = []
    n_jump = 0   # number of "saw event but no frame"
    while (trig_idx < len(triggers)) and (raw_idx < len(raw_ts)):
        trig_t, raw_t = triggers[trig_idx], raw_ts[raw_idx]
        if np.abs(trig_t - raw_t) < thresh:
            new_trigs.append(trig_t)
            n_jump = 0

            trig_idx += 1
            raw_idx += 1
        else:
            n_jump += 1
            trig_idx += 1
            assert n_jump < 2, "more than 2 frames jumped!! code does not handel such a case!!"
    
    return np.array(new_trigs)


def main(trig_f, raw_dir, thresh=4500):
    # trig_f = "work_dir/black_seoul_b3_v3/triggers.txt"
    # raw_dir = "Videos/black_seoul_b3_v3"
    # thresh = 4500.0
    

    backup_f = osp.dirname(trig_f) + "backup_" + osp.basename(trig_f)
    if not osp.exists(backup_f):
        print("making trigger backup")
        shutil.copy(trig_f, backup_f)
    else:
        print("backup already exists")

    raw_fs = sorted(glob.glob(osp.join(raw_dir, "*.raw")))
    raw_ts = parse_raw_ts(raw_fs)
    triggers = np.loadtxt(trig_f)

    fixed_triggers = fix_trigger(triggers, raw_ts, thresh=thresh)
    np.savetxt(trig_f, fixed_triggers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trig_f", help="path to trigger", default=None)
    parser.add_argument("--raw_dir", help="path to directory of raw files", default=None)
    parser.add_argument("--thresh", help="acceptible error considered the triggers is correct", default=4500)
    args = parser.parse_args()


    main(args.trig_f, args.raw_dir, args.thresh)