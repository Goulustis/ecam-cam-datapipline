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
        cam_json["t"] = t
        with open(targ_cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)


def write_metadata(eimgs_ids, targ_dir):
    """
    saves the eimg ids as metatdata 
    input:
        eimgs_ids (np.array [int]) : event image ids
        targ_dir (str): directory to save to
    """
    metadata = {}

    for i, id in enumerate(eimgs_ids):
        metadata[str(i).zfill(6)] = {"warp_id":id,
                                     "appearence_id":id,
                                     "camera_id":0}
    
    with open(osp.join(targ_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)



def format_ecam_data(data_path, ecam_intrinsics_path, targ_dir, trig_path):
    os.makedirs(targ_dir, exist_ok=True)
    event_path = osp.join(data_path, "processed_event.h5") 
    ecam_path = osp.join(data_path, "e_cams.npy")  ## extrinsics

    # read files
    ecams_trig = np.load(ecam_path) # extrinsics at trigger times
    intr_mtx, dist = read_ecam_intrinsics(ecam_intrinsics_path)
    triggers = read_triggers(trig_path)

    ## create event images
    events = read_events(event_path, save_np=True, targ_dir=targ_dir)
    eimgs, eimg_ts, eimgs_ids, trig_ids = create_event_imgs(events, triggers)
    
    ecams = create_interpolated_ecams(eimg_ts, triggers, ecams_trig)

    ## create nerfies.Camera and save extrinsics
    extrinsic_dir = osp.join(targ_dir, "camera")
    create_camera_extrinsics(extrinsic_dir, ecams, eimg_ts, intr_mtx, dist)

    # create metadata.json
    write_metadata(eimgs_ids, targ_dir)

    # save the event images
    np.save(osp.join(targ_dir, "eimgs.npy"), eimgs)

    # save the trig_ids; make the color camera ids the same
    np.save(osp.join(targ_dir, "trig_ids.npy"), trig_ids)

    # copy event to places
    shutil.copyfile(event_path, targ_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make the event camera extrinsics dataset")

    # data_path is for getting the event h5 and the event camera extrinsics
    parser.add_argument("--data_path", help="the path to the dataset format described in readme", default="data/checker")
    parser.add_argument("--ecam_int_path", help="path to rel_cam.json", default="data/checker/rel_cam.json")
    parser.add_argument("--targ_dir", help="location to save the formatted dataset", default="data/formatted_ecam_checker")
    parser.add_argument("--trigger_path", help="path to ecam triggers", default="data/checker/triggers.txt")
    # parser.add_argument("--scale",type=int, help="factor to scale the extrinsics by, since color camera could use lower res", default=2) # will not happen, because no change for event camera extrinsics
    args = parser.parse_args()

    format_ecam_data(args.data_path, args.ecam_int_path, args.targ_dir, args.trigger_path)