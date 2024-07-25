import cv2
import argparse
import os 
import os.path as osp
from tqdm import tqdm
import numpy as np
import shutil
import glob
import json
import sys

from extrinsics_visualization.colmap_scene_manager import ColmapSceneManager
from format_data.format_utils import EventBuffer, read_triggers
from format_data.eimg_maker import ev_to_eimg
from format_data.slerp_qua import CameraSpline
from format_data.format_utils import read_triggers, read_ecam_intrinsics
from format_data.format_ecam_set import create_and_write_camera_extrinsics, calc_t_shift
from format_data.slerp_qua import create_interpolated_ecams
from extrinsics_creator.create_rel_cam import apply_rel_cam

# def create_eimg_by_triggers(events, triggers, exposure_time = 5000, make_eimg=True):
def create_eimg_by_triggers(events:EventBuffer, triggers, exposure_time = 14980, make_eimg=True):
    eimgs = np.zeros((len(triggers), 720, 1280), dtype=np.uint8)
    eimg_ts = []
    for i, trigger in tqdm(enumerate(triggers), total=len(triggers), desc="making ev imgs"):
        # st_t, end_t = max(trigger - exposure_time//2, 0), trigger + exposure_time//2
        st_t, end_t = trigger, trigger + exposure_time

        if st_t > events.t_f[-1]:
            break

        eimg_ts.append(st_t + exposure_time//2)

        if make_eimg:
            curr_t, curr_x, curr_y, curr_p = events.retrieve_data(st_t, end_t)

            eimg = ev_to_eimg(curr_x, curr_y, curr_p).astype(np.uint8)
            eimg[eimg != 0] = 255
            eimgs[i] = eimg
            
            events.drop_cache_by_t(st_t)
    
    return eimgs, np.array(eimg_ts)


def load_scale_factor(ev_f):
    work_dir = osp.dirname(ev_f)
    scale_f = osp.join(work_dir, f"{osp.basename(work_dir)}_recon", "colmap_scale.txt")
    with open(scale_f, "r") as f:
        return float(f.read())


if __name__ == "__main__":
    MAKE_EIMG=True
    scene = "board_v13_t2_c2"
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input event file", default=f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{scene}/processed_events.h5")
    parser.add_argument("-t", "--trigger_f", help="path to trigger.txt file, expect start_trigger.txt", default=f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{scene}/triggers.txt")
    parser.add_argument("-o", "--output", help="output directory", default=f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{scene}/trig_eimgs")
    parser.add_argument("-w", "--workdir", help="directory used to create the scene", default=f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{scene}")
    # parser.add_argument("-w", "--workdir", help="directory used to create the scene", default=None)

    # parser.add_argument("-i", "--input", help="input event file", default=f"/scratch-ssd/workdir/{scene}/processed_events.h5")
    # parser.add_argument("-t", "--trigger_f", help="path to trigger.txt file", default=f"/scratch-ssd/workdir/{scene}/triggers.txt")
    # parser.add_argument("-o", "--output", help="output directory", default=f"/scratch-ssd/workdir/{scene}/trig_eimgs")
    # parser.add_argument("-w", "--workdir", help="directory used to create the scene", default=f"/scratch-ssd/workdir/{scene}")
    args = parser.parse_args()
    scene = osp.basename(args.workdir)

    events = EventBuffer(args.input)
    triggers = read_triggers(args.trigger_f)
    manager = ColmapSceneManager(osp.join(args.workdir, f"{scene}_recon"))
    colcam_extrinsics = manager.get_all_extrnxs()
    # triggers = triggers[manager.get_found_cond(len(triggers))]

    os.makedirs(args.output, exist_ok=True)

    t_shift = calc_t_shift(args.trigger_f)
    try:
        eimgs, eimg_ts = create_eimg_by_triggers(events, triggers + t_shift, exposure_time=30000, make_eimg=MAKE_EIMG)
    except Exception as e:
        shutil.rmtree(args.output)
        print(e)
        assert 0

    if MAKE_EIMG:
        for i, eimg in tqdm(enumerate(eimgs), total=len(eimgs), desc="saving eimgs"):
            save_f = osp.join(args.output, f"{str(i).zfill(6)}.png")
            write_flag = cv2.imwrite(save_f, eimg)
            assert write_flag, "write image failed"
    
    if args.workdir is not None:
        ## make cameras
        try:
            SCALE=load_scale_factor(args.input)
        except FileNotFoundError as e:
            print(e)
            print("This is fine if manual calibrating")
            sys.exit(1)
        trig_ecam_set_dir = osp.join(osp.dirname(args.output), "trig_ecamset")
        eimgs_dir = osp.join(trig_ecam_set_dir, "eimgs")
        trig_ecam_dir = osp.join(trig_ecam_set_dir, "camera")

        os.makedirs(eimgs_dir, exist_ok=True)
        os.makedirs(trig_ecam_set_dir, exist_ok=True)

        if MAKE_EIMG:
            eimgs = eimgs[manager.get_found_cond(len(triggers))]
            np.save(osp.join(eimgs_dir, "eimgs_1x.npy"), eimgs)

        
        with open(osp.join(args.workdir, "rel_cam.json"), "r") as f:
            rel_cam = json.load(f)
            rel_cam["R"], rel_cam["T"] = np.array(rel_cam["R"]), np.array(rel_cam["T"])

        
        ecams = apply_rel_cam(rel_cam, colcam_extrinsics, SCALE)
        # ecams = create_interpolated_ecams(eimg_ts, triggers + t_shift, ecams)

        create_and_write_camera_extrinsics(trig_ecam_dir, ecams, eimg_ts, np.array(rel_cam["M2"]), np.array(rel_cam["dist2"]))
        print("saved trig ecamset to:", trig_ecam_set_dir)
        
