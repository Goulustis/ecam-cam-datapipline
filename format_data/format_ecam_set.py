import numpy as np
import json
import os
import os.path as osp
from nerfies.camera import Camera
import argparse
import shutil
from utils import read_triggers

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

def read_ecam_intrinsics(path, cam_i = 2):
    """
    input:
        path (str): path to json
    output:
        M (np.array): 3x3 intrinsic matrix
        dist (list like): distortion (k1, k2, p1, p2, k3)
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    return np.array(data[f"M{cam_i}"]), data[f"dist{cam_i}"][0]


def create_camera_extrinsics(extrinsic_dir, ecams, triggers, intr_mtx, dist):
    os.makedirs(extrinsic_dir, exist_ok=True)
    for i, (ecam,t) in enumerate(zip(ecams, triggers)):
        camera = make_camera(ecam, intr_mtx, dist)
        targ_cam_path = osp.join(extrinsic_dir, str(i).zfill(6) + ".json")
        print("saveing to", targ_cam_path)
        cam_json = camera.to_json()
        cam_json["t"] = t
        with open(targ_cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)


def format_ecam_data(data_path, ecam_intrinsics_path, targ_dir, ts_path):
    os.makedirs(targ_dir, exist_ok=True)
    event_path = osp.join(data_path, "processed_event.h5") 
    ecam_path = osp.join(data_path, "e_cams.npy")  ## extrinsics

    # read files
    ecams = np.load(ecam_path)
    intr_mtx, dist = read_ecam_intrinsics(ecam_intrinsics_path)
    ecam_times = read_triggers(ts_path)

    ## create camera extrinsics
    extrinsic_dir = osp.join(targ_dir, "camera")
    format_ecam_data(extrinsic_dir, ecams, ecam_times, intr_mtx, dist)

    # create event images according to extrinsic time stamp

    # create metadata.json
    
    # copy event to places
    shutil.copyfile(event_path, targ_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make the event camera extrinsics dataset")

    # data_path is for getting the event h5 and the event camera extrinsics
    parser.add_argument("--data_path", help="the path to the dataset format described in readme", default="data/checker")
    parser.add_argument("--ecam_int_path", help="path to rel_cam.json", default="data/checker/rel_cam.json")
    parser.add_argument("--targ_dir", help="location to save the formatted dataset", default="data/formatted_ecam_checker")
    parser.add_argument("--cam_t_path", help="path to ecam extrinsic time stamps", default="data/checker/triggers.txt")
    # parser.add_argument("--scale",type=int, help="factor to scale the extrinsics by, since color camera could use lower res", default=2) # will not happen, because no change for event camera extrinsics
    args = parser.parse_args()

    format_ecam_data(args.data_path, args.ecam_int_path, args.targ_dir, args.cam_t_path)