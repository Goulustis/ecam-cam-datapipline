import numpy as np
import json
import os
import os.path as osp
from nerfies.camera import Camera
import argparse
import shutil


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
    t = ext_mtx[:3,3][None] 
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
        image_size=np.array([1280, 720])
    )

    return new_camera

def read_ecam_intrinsics(path):
    """
    input:
        path (str): path to json
    output:
        M (np.array): 3x3 intrinsic matrix
        dist (list like): distortion (k1, k2, p1, p2, k3)
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    return np.array(data["M2"]), data["dist2"]


def format_ecam_data(data_path, ecam_intrinsics_path, targ_dir):
    os.makedirs(targ_dir, exist_ok=True)
    event_path = osp.join(data_path, "processed_event.h5") 
    ecam_path = osp.join(data_path, "e_cams.npy")  ## extrinsics

    # read files
    ecams = np.load(ecam_path)
    intr_mtx, dist = read_ecam_intrinsics(ecam_intrinsics_path)

    ## create cameras
    extrinsic_dir = osp.join(targ_dir, "camera")
    os.makedirs(extrinsic_dir, exist_ok=True)
    for i, ecam in enumerate(ecams):
        camera = make_camera(ecam, intr_mtx, dist)
        targ_cam_path = osp.join(extrinsic_dir, str(i).zfill(6) + ".json")
        print("saveing to", targ_cam_path)
        with open(targ_cam_path, "w") as f:
            json.dump(camera.to_json(), f, indent=2)
    
    # copy event to places
    shutil.copyfile(event_path, targ_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make the event camera extrinsics dataset")
    parser.add_argument("--data_path", help="the path to the dataset format described in readme", default="data/checker")
    parser.add_argument("--ecam_int_path", help="path to rel_cam.json", default="data/checker/rel_cam.json")
    parser.add_argument("--targ_dir", help="location to save the formatted dataset", default="data/formatted_ecam_checker")
    parser.add_argument("--scale",type=int, help="factor to scale the extrinsics by, since color camera could use lower res", default=2)
    args = parser.parse_args()

    format_ecam_data(args.data_path, args.ecam_int_path, args.targ_dir)