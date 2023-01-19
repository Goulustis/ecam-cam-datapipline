import numpy as np
import json
import os
import os.path as osp
from nerfies.camera import Camera
import argparse
import shutil

from utils import read_triggers, read_ecam_intrinsics, read_events
from eimg_maker import create_event_imgs
from slerp_qua import create_interpolated_ecams

# TODO: step throught is function for sanity check
def make_camera(ext_mtx, intr_mtx, dist):
    """
    input:
        ext_mtx (np.array): World to cam matrix - shape = 4x4
        intr_mtx (np.array): intrinsic matrix of camera - shape = 3x3

    return:
        nerfies.camera.Camera of the given mtx
    """
    R = ext_mtx[:3,:3]
    t = ext_mtx[:3,3]
    k1, k2, p1, p2, k3 = dist
    coord = -t.T@R  # step to check this is correct

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


def create_camera_extrinsics(extrinsic_dir, ecams, triggers, intr_mtx, dist):
    """
    create the extrinsics and save it
    """
    os.makedirs(extrinsic_dir, exist_ok=True)
    for i, (ecam,t) in enumerate(zip(ecams, triggers)):
        camera = make_camera(ecam, intr_mtx, dist)
        targ_cam_path = osp.join(extrinsic_dir, str(i).zfill(6) + ".json")
        print("saving to", targ_cam_path)
        cam_json = camera.to_json()
        cam_json["t"] = int(t)
        with open(targ_cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)


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
    

def format_ecam_data(data_path, ecam_intrinsics_path, targ_dir, trig_path):
    os.makedirs(targ_dir, exist_ok=True)
    event_path = osp.join(data_path, "processed_events.h5") 
    ecam_path = osp.join(data_path, "e_cams.npy")  ## trigger extrinsics

    # read files
    ecams_trig = np.load(ecam_path) # extrinsics at trigger times
    intr_mtx, dist = read_ecam_intrinsics(ecam_intrinsics_path)
    triggers = read_triggers(trig_path)

    ## create event images
    events = read_events(event_path, save_np=True, targ_dir=targ_dir)
    # events = None
    eimgs, eimg_ts, eimgs_ids, trig_ids = create_event_imgs(events, triggers, create_imgs=True)

    save_eimgs(eimgs, targ_dir)
    
    ecams = create_interpolated_ecams(eimg_ts, triggers, ecams_trig)

    ## create nerfies.Camera and save extrinsics
    extrinsic_dir = osp.join(targ_dir, "camera")
    create_camera_extrinsics(extrinsic_dir, ecams, eimg_ts, intr_mtx, dist)

    # create metadata.json
    write_metadata(eimgs_ids, eimg_ts, targ_dir)

    # save the trig_ids; make the color camera ids the same
    np.save(osp.join(targ_dir, "trig_ids.npy"), trig_ids)

    # copy event to places
    shutil.copyfile(event_path, osp.join(targ_dir, osp.basename(event_path)))

    # create train valid split
    write_train_valid_split(eimgs_ids, targ_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make the event camera extrinsics dataset")

    # data_path is for getting the event h5 and the event camera extrinsics
    parser.add_argument("--scene_path", help="the path to the dataset format described in readme", default="data/checker")
    parser.add_argument("--relcam_path", help="path to rel_cam.json containing relative camera info", default="data/checker/rel_cam.json")
    parser.add_argument("--targ_dir", help="location to save the formatted dataset", default="data/formatted_checker/ecam_set")
    parser.add_argument("--trigger_path", help="path to ecam triggers", default="data/checker/triggers.txt")
    args = parser.parse_args()

    format_ecam_data(args.scene_path, args.relcam_path, args.targ_dir, args.trigger_path)