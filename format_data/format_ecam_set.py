import sys
sys.path.append(".")

import numpy as np
import json
import os
import os.path as osp
from nerfies.camera import Camera
import argparse
import shutil
import warnings

from format_data.format_utils import read_triggers, read_ecam_intrinsics, read_events, EventBuffer
from format_data.eimg_maker import create_event_imgs
from format_data.slerp_qua import create_interpolated_ecams
from extrinsics_visualization.colmap_scene_manager import ColmapSceneManager


def make_camera(ext_mtx, intr_mtx, dist):
    """
    input:
        ext_mtx (np.array): World to cam matrix - shape = 4x4
        intr_mtx (np.array): intrinsic matrix of camera - shape = 3x3

    return:
        nerfies.camera.Camera of the given mtx
    """
    dist = dist.squeeze()
    R = ext_mtx[:3,:3]
    t = ext_mtx[:3,3]
    k1, k2, p1, p2, k3 = dist
    coord = -t.T@R  

    cx, cy = intr_mtx[:2,2].astype(int)

    new_camera = Camera(
        orientation=R,
        position=coord,
        focal_length=intr_mtx[0,0],
        pixel_aspect_ratio=1,
        principal_point=np.array([cx, cy]),
        radial_distortion=(k1, k2, 0),
        tangential_distortion=(p1, p2),
        skew=0,
        image_size=np.array([1280, 720])  ## (width, height) of camera
    )

    return new_camera


def create_and_write_camera_extrinsics(extrinsic_dir, ecams, triggers, intr_mtx, dist, ret_cam=False):
    """
    create the extrinsics and save it
    """
    os.makedirs(extrinsic_dir, exist_ok=True)
    cameras = []
    for i, (ecam,t) in enumerate(zip(ecams, triggers)):
        camera = make_camera(ecam, intr_mtx, dist)
        cameras.append(camera)
        targ_cam_path = osp.join(extrinsic_dir, str(i).zfill(6) + ".json")
        print("saving to", targ_cam_path)
        cam_json = camera.to_json()
        cam_json["t"] = int(t)
        with open(targ_cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)

    if ret_cam:
        return cameras

def write_metadata(eimgs_ids, eimgs_ts, targ_dir):
    """
    saves the eimg ids as metatdata 
    input:
        eimgs_ids (np.array [int]) : event image ids
        eimgs_ts (np.array [int]) : time stamp of each event image
        targ_dir (str): directory to save to
    """
    metadata = {}

    for i, (id, t) in enumerate(zip(eimgs_ids, eimgs_ts)):
        metadata[str(i).zfill(6)] = {"warp_id":int(id),
                                     "appearance_id":int(id),
                                     "camera_id":0,
                                     "t":int(t)}
    
    with open(osp.join(targ_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def write_train_valid_split(eimgs_ids, targ_dir):
    eimgs_ids = [str(int(e)).zfill(6) for e in eimgs_ids]
    save_path = osp.join(targ_dir, "dataset.json")

    train_ids = sorted(eimgs_ids)
    dataset_json = {
        "count":len(eimgs_ids),
        "num_exemplars":len(train_ids),
        "train_ids": eimgs_ids,
        "val_ids":[]
    }

    with open(save_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

def save_eimgs(eimgs, targ_dir):
    if eimgs is None:
        return 

    eimgs_dir = osp.join(targ_dir, "eimgs")
    os.makedirs(eimgs_dir, exist_ok=True)
    np.save(osp.join(eimgs_dir, "eimgs_1x.npy"), eimgs)
    del eimgs
    

def calc_t_shift(trig_path):
    st_trig_f = osp.join(osp.dirname(trig_path), "st_triggers.txt")
    end_trig_f = osp.join(osp.dirname(trig_path), "end_triggers.txt")
    if osp.exists(st_trig_f):
        st_trigs, end_trigs = np.loadtxt(st_trig_f), np.loadtxt(end_trig_f)
        return (end_trigs[0] - st_trigs[0])/2
    
    warnings.warn("not shifting to mean exposure time stamp, events might drift")
    return 0


def calc_time_delta(triggers, min_mult=3):
    delta_t = triggers[1] - triggers[0]
    n_mult = np.round(delta_t/5000)
    n_mult = max(min_mult, n_mult)
    
    return np.ceil(delta_t/n_mult).astype(int)


def format_ecam_data(data_path, ecam_intrinsics_path, targ_dir, trig_path, create_eimgs, colmap_dir):
    os.makedirs(targ_dir, exist_ok=True)
    event_path = osp.join(data_path, "processed_events.h5") 
    ecam_path = osp.join(data_path, "e_cams.npy")  ## trigger extrinsics

    # read files
    manager = ColmapSceneManager(colmap_dir)
    trig_ecams = np.load(ecam_path) # extrinsics at trigger times
    intr_mtx, dist = read_ecam_intrinsics(ecam_intrinsics_path)
    triggers = read_triggers(trig_path)
    cond = manager.get_found_cond(len(triggers))
    triggers = triggers[cond]

    ## create event images
    events = EventBuffer(event_path)
    time_delta = calc_time_delta(triggers)
    # eimgs, eimg_ts, eimgs_ids, trig_ids = create_event_imgs(events, triggers, create_imgs=(create_eimgs=="True"), time_delta=5100)
    eimgs, eimg_ts, eimgs_ids, trig_ids = create_event_imgs(events, triggers, create_imgs=(create_eimgs=="True"), time_delta=time_delta)

    save_eimgs(eimgs, targ_dir)

    t_shift = calc_t_shift(trig_path)
    # ecams = create_interpolated_ecams(eimg_ts, triggers + 7490, trig_ecams)  ## debug check later
    ecams = create_interpolated_ecams(eimg_ts, triggers + t_shift, trig_ecams)

    ## create nerfies.Camera and save extrinsics
    extrinsic_targ_dir = osp.join(targ_dir, "camera")
    create_and_write_camera_extrinsics(extrinsic_targ_dir, ecams, eimg_ts, intr_mtx, dist)

    # create metadata.json
    write_metadata(eimgs_ids, eimg_ts, targ_dir)

    # save the trig_ids; make the color camera ids the same
    np.save(osp.join(targ_dir, "trig_ids.npy"), trig_ids)

    # copy event to places
    # shutil.copyfile(event_path, osp.join(targ_dir, osp.basename(event_path)))

    # create train valid split
    write_train_valid_split(eimgs_ids, targ_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make the event camera extrinsics dataset")

    # data_path is for getting the event h5 and the event camera extrinsics
    # parser.add_argument("--scene_path", help="the path to the dataset format described in readme", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/office_b2_v3")
    # parser.add_argument("--relcam_path", help="path to rel_cam.json containing relative camera info", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/office_b2_v3/rel_cam.json")
    # parser.add_argument("--trigger_path", help="path to ecam triggers only rgb open shutter ones", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/office_b2_v3/triggers.txt")

    dataset = "grad_lounge_b2_v1"
    parser.add_argument("--scene_path", help="the path to the dataset format described in readme", default=f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{dataset}")
    parser.add_argument("--relcam_path", help="path to rel_cam.json containing relative camera info", default=f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{dataset}/rel_cam.json")
    parser.add_argument("--trigger_path", help="path to ecam triggers only rgb open shutter ones", default=f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{dataset}/triggers.txt")
    parser.add_argument("--colmap_dir", help="directory to xxx_recons")

    parser.add_argument("--targ_dir", help="location to save the formatted dataset", default="debug")
    parser.add_argument("--create_eimgs", choices=["True", "False"], default="True")
    args = parser.parse_args()

    format_ecam_data(args.scene_path, args.relcam_path, args.targ_dir, args.trigger_path, args.create_eimgs, args.colmap_dir)