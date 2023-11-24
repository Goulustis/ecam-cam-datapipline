import cv2
import argparse
import os 
import os.path as osp
from tqdm import tqdm
import numpy as np
import shutil
import glob
import json

from extrinsics_visualization.colmap_scene_manager import ColSceneManager
from format_data.utils import EventBuffer, read_triggers
from format_data.eimg_maker import ev_to_img
from format_data.slerp_qua import CameraSpline
from format_data.utils import read_triggers, read_ecam_intrinsics
from format_data.format_ecam_set import create_and_write_camera_extrinsics
from extrinsics_creator.create_rel_cam import map_cam

def create_eimg_by_triggers(events, triggers, exposure_time = 5000):
    eimgs = np.zeros((len(triggers), 720, 1280), dtype=np.uint8)
    eimg_ts = []
    for i, trigger in tqdm(enumerate(triggers), total=len(triggers), desc="making ev imgs"):
        st_t, end_t = max(trigger - exposure_time//2, 0), trigger + exposure_time//2
        eimg_ts.append(st_t )
        curr_t, curr_x, curr_y, curr_p = events.retrieve_data(st_t, end_t)

        eimg = ev_to_img(curr_x, curr_y, curr_p)
        eimg[eimg != 0] = 255
        eimgs[i] = eimg
        
        events.drop_cache_by_t(st_t)
    
    return eimgs, eimg_ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input event file", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/processed_events.h5")
    parser.add_argument("-t", "--trigger_f", help="path to trigger.txt file", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/triggers.txt")
    parser.add_argument("-o", "--output", help="output directory", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/trig_eimgs")
    parser.add_argument("-w", "--workdir", help="directory used to create the scene", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon")
    # parser.add_argument("-w", "--workdir", help="directory used to create the scene", default=None)
    args = parser.parse_args()

    events = EventBuffer(args.input)
    triggers = read_triggers(args.trigger_f)

    os.makedirs(args.output, exist_ok=True)

    try:
        eimgs, eimg_ts = create_eimg_by_triggers(events, triggers,10000)
    except Exception as e:
        shutil.rmtree(args.output)
        print(e)
        assert 0

    for i, eimg in tqdm(enumerate(eimgs), total=len(eimgs), desc="saving eimgs"):
        save_f = osp.join(args.output, f"{str(i).zfill(6)}.png")
        write_flag = cv2.imwrite(save_f, eimg)
        assert write_flag, "write image failed"
    

    if args.workdir is not None:
        trig_ecam_set_dir = osp.join(osp.dirname(args.output), "trig_ecamset")
        eimgs_dir = osp.join(trig_ecam_set_dir, "eimgs")
        trig_ecam_dir = osp.join(trig_ecam_set_dir, "camera")

        os.makedirs(eimgs_dir, exist_ok=True)
        os.makedirs(trig_ecam_set_dir, exist_ok=True)

        np.save(osp.join(eimgs_dir, "eimgs_1x.npy"), eimgs)

        manager = ColSceneManager(glob.glob(osp.join(args.workdir, "*recon*"))[0])
        colcam_extrinsics = [manager.get_extrnxs(idx) for idx in sorted(list(manager.images.keys()))]
        
        with open(osp.join(args.workdir, "rel_cam.json"), "r") as f:
            rel_cam = json.load(f)
            rel_cam["R"], rel_cam["T"] = np.array(rel_cam["R"]), np.array(rel_cam["T"])
        ecams = map_cam(rel_cam, colcam_extrinsics, 0.158)

        create_and_write_camera_extrinsics(trig_ecam_dir, ecams, eimg_ts, np.array(rel_cam["M2"]), np.array(rel_cam["dist2"]))
        print("saved trig ecamset to:", trig_ecam_set_dir)
        
