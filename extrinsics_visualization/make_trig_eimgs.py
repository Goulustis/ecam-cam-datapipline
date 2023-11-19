import cv2
import argparse
import os 
import os.path as osp
from tqdm import tqdm
import numpy as np
import shutil

from format_data.utils import EventBuffer, read_triggers
from format_data.eimg_maker import ev_to_img


def create_eimg_by_triggers(events, triggers, exposure_time = 5000):
    eimgs = np.zeros((len(triggers), 720, 1280), dtype=np.uint8)
    for i, trigger in tqdm(enumerate(triggers), total=len(triggers), desc="making ev imgs"):
        st_t, end_t = max(trigger - exposure_time//2, 0), trigger + exposure_time//2
        curr_t, curr_x, curr_y, curr_p = events.retrieve_data(st_t, end_t)

        eimg = ev_to_img(curr_x, curr_y, curr_p)
        eimg[eimg != 0] = 255
        eimgs[i] = eimg
        
        events.drop_cache_by_t(st_t)
    
    return eimgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input event file", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1/processed_events.h5")
    parser.add_argument("-t", "--trigger_f", help="path to trigger.txt file", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1/triggers.txt")
    parser.add_argument("-o", "--output", help="output directory", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1/trig_eimgs")
    args = parser.parse_args()

    events = EventBuffer(args.input)
    triggers = read_triggers(args.trigger_f)

    os.makedirs(args.output, exist_ok=True)

    try:
        eimgs = create_eimg_by_triggers(events, triggers)
    except Exception as e:
        shutil.rmtree(args.output)
        print(e)
        assert 0

    for i, eimg in tqdm(enumerate(eimgs), total=len(eimgs), desc="saving eimgs"):
        save_f = osp.join(args.output, f"{str(i).zfill(4)}.png")
        write_flag = cv2.imwrite(save_f, eimg)
        assert write_flag, "write image failed"
    


    
