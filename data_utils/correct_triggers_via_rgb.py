import numpy as np
import argparse
import os.path as osp
import glob
import shutil

import bisect



def calculate_microseconds_from_filename(time_str):
    # Extract the time and microseconds parts from the filename
    parts = time_str.split('_')
    time_part = parts[3]  # HHMMSS
    microseconds_part = parts[4]  # <microseconds> without ".raw"

    # Convert HHMMSS to microseconds
    hours = int(time_part[:2])
    minutes = int(time_part[2:4])
    seconds = int(time_part[4:6])
    
    total_microseconds = (hours * 3600 + minutes * 60 + seconds) * 1000000 + int(microseconds_part)
    
    return total_microseconds


def to_microsec(time_str):
    """
    Converts a time string in the format "hh_mm_ss_mmm" to microseconds.
    Here, hh = hours, mm = minutes, ss = seconds, mmm = milliseconds.
    """
    if "raw_image" in time_str:
        return calculate_microseconds_from_filename(time_str)
        
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


def find_nearest_elements_with_threshold(list1, list2, threshold):
    nearest_elements = []
    for num in list2:
        pos = bisect.bisect_left(list1, num)
        nearest = None
        # Find the nearest by comparing the neighbors
        if pos == 0:
            nearest = list1[0]
        elif pos == len(list1):
            nearest = list1[-1]
        else:
            # Choose the nearest between list1[pos] and list1[pos-1]
            prev_diff = abs(list1[pos-1] - num)
            next_diff = abs(list1[pos] - num)
            if prev_diff <= next_diff:
                nearest = list1[pos-1]
            else:
                nearest = list1[pos]

        # Check if the nearest element is within the threshold
        if abs(nearest - num) > threshold:
            nearest_elements.append(-1)
        else:
            nearest_elements.append(nearest)
            
    return nearest_elements



def fix_trigger(triggers:np.ndarray, raw_ts:np.ndarray, thresh:float=4500):
    raw_ts = raw_ts - raw_ts[0] + triggers[0]

    t_delta = raw_ts[1] - raw_ts[0]
    trig_hypo_raw = np.array(list(range(len(raw_ts)))) * t_delta

    raw_diff_diff = np.concatenate([np.zeros(1), np.diff(raw_ts) - t_delta])
    n_add = np.cumsum(np.round(raw_diff_diff/(1.8*t_delta)))
    trig_hypo = trig_hypo_raw + n_add * t_delta + triggers[0]
    new_trigs = np.array(find_nearest_elements_with_threshold(triggers, trig_hypo, thresh))
    # new_trigs = sorted(new_trigs)
    new_trigs = new_trigs[new_trigs >= 0]

    return new_trigs


def main(trig_f, raw_dir, thresh=4500):
    # trig_f = "work_dir/black_seoul_b3_v3/triggers.txt"
    # raw_dir = "Videos/black_seoul_b3_v3"
    # thresh = 4500.0
    

    backup_f = osp.join(osp.dirname(trig_f) , "backup_" + osp.basename(trig_f))
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
    parser.add_argument("--trig_f", help="path to trigger", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/playground_v6/triggers.txt")
    parser.add_argument("--raw_dir", help="path to directory of raw files", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/playground_v6")
    parser.add_argument("--thresh", help="acceptible error considered the triggers is correct in microseconds", default=4500)
    args = parser.parse_args()


    main(args.trig_f, args.raw_dir, args.thresh)