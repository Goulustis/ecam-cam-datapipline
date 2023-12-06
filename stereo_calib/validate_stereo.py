from tqdm import tqdm
import json
import cv2
import numpy as np
import glob
import os.path as osp
import os
import matplotlib.pyplot as plt
import contextlib

from extrinsics_visualization.colmap_scene_manager import ColSceneManager, proj_3d_pnts
from extrinsics_correction.point_selector import ImagePointSelector
from extrinsics_correction.manual_scale_finding import pnp_extrns, triangulate_points
from extrinsics_creator.create_rel_cam import map_cam
from utils.misc import parallel_map
from utils.images import calc_clearness_score

TMP_DIR = "./tmp"


def get_img_fs(rgb_dir, eimg_dir, n_use = 300, st_n=150):
    sub_sample = lambda x : x[::len(x)//n_use]

    images_to = sorted(glob.glob(osp.join(rgb_dir, "*")))[st_n:]
    images_from = sorted(glob.glob(osp.join(eimg_dir, "*")))[st_n:]

    n_frames = min(len(images_from), len(images_to))
    images_from = images_from[:n_frames]
    images_to = images_to[:n_frames]

    images_to, images_from = sub_sample(images_to), sub_sample(images_from)

    return images_to, images_from

def get_calib_objpnts(grid_size=3):
    row, col = 5,8   # inner cornors of the checker, see https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
    objp = np.zeros((row*col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)
    objp = objp*grid_size

    return objp

def find_all_extrnsics(imgs, K, D, pnts_2d=None):
    objpnts = get_calib_objpnts(3)

    pnts_2d = parallel_map(lambda x: cv2.findChessboardCorners(x, (5, 8), None), imgs, show_pbar=True, desc="finding corners") #(r, c)
    cond = np.stack(e[0] for e in pnts_2d)
    pnts_2d = np.stack([e[1] for e in pnts_2d])[cond]


    extrs = []
    for pnt in pnts_2d:
        R, t = pnp_extrns(objpnts, pnt, K, D)
        extrs.append(np.concatenate([R, t], axis=1))
    
    return extrs, cond


def load_relcam(relcam_f):
    with open(relcam_f, "r") as f:
        data = json.load(f)
    
    return np.array(data["M1"]), np.array(data["dist1"]), np.array(data["M2"]), np.array(data["dist2"]), np.array(data["R"]), np.array(data["T"])


def stereo_rectify_and_overlay(image1, image2, K1, D1, K2, D2, R, T, imageSize):
    # Stereo rectify
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, imageSize, R, T)

    # Compute rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, imageSize, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, imageSize, cv2.CV_32FC1)

    # Rectify images
    rectified_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR)
    rectified_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR)

    # Create an overlay image
    overlay_img = np.zeros_like(rectified_img1)
    # overlay_img[..., 0] = rectified_img2[..., 0]  # Blue channel from image2
    # overlay_img[..., 2] = rectified_img1[..., 2]  # Red channel from image1

    overlay_img[..., 0] = cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2GRAY)  # Blue channel from image2
    overlay_img[..., 2] = cv2.cvtColor(rectified_img2, cv2.COLOR_BGR2GRAY)  # Red channel from image1

    return overlay_img



def validate_in_stereo_space():
    rgb_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/calib_checker_recons/images"
    eimg_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/ecam_code/raw_events/calib_checker/events_imgs"
    relcam_f = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/ecam_code/raw_events/calib_checker/rel_cam.json"
    save_dir = osp.join(TMP_DIR, "stereo_space_reproj")

    used_idxs = sorted([int(osp.basename(e).split(".")[0]) for e in glob.glob(osp.join(TMP_DIR, "stereo_calib_imgs", "*.png"))])
    objpnts = get_calib_objpnts()

    select_fn = lambda x, cond : [x[e] for e in cond]
    rgb_fs = sorted(glob.glob(osp.join(rgb_dir, "*.png")))
    eimg_fs = sorted(glob.glob(osp.join(eimg_dir, "*.png")))
    rgb_fs = rgb_fs[:len(eimg_fs)]
    # rgb_fs, eimg_fs = get_img_fs(rgb_dir, eimg_dir)
    # rgb_fs, eimg_fs = select_fn(rgb_fs, used_idxs), select_fn(eimg_fs, used_idxs)
    
    rgb_K, rgb_D, ecam_K, ecam_D, R, T = load_relcam(relcam_f)

    rgbs, eimgs = parallel_map(cv2.imread, rgb_fs, show_pbar=True, desc="loading rgbs"), parallel_map(cv2.imread, eimg_fs, show_pbar=True, desc="loading eimgs")
    rgb_cams, cond = find_all_extrnsics(rgbs, rgb_K, rgb_D)

    eimgs = [e for e, c in zip(eimgs, cond) if c]
    ecams = map_cam({"R":R, "T":T}, rgb_cams, scale=1)

    def proj_fn(inp):
        img, extr = inp
        return proj_3d_pnts(img, ecam_K, extr, objpnts, dist_coeffs=ecam_D)[1]

    proj_eimgs = parallel_map(proj_fn, list(zip(eimgs, ecams)))
    os.makedirs(save_dir, exist_ok=True)
    
    def save_fn(inp):
        img, idx = inp
        cv2.imwrite(osp.join(save_dir, f"{str(idx).zfill(6)}.png"), img)
    
    parallel_map(save_fn, list(zip(proj_eimgs, [e for (e,c) in zip(used_idxs, cond) if c])), 
                 show_pbar=True, desc="saving projected")
    


def find_chessboard_corners(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, pnts = cv2.findChessboardCorners(img, (5,8), None)
    assert ret

    criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(img, pnts, (11,11),(-1,-1), criteria)
    return pnts

def load_objpnts(colmap_pnts_f, colmap_dir=None, calc_clear=False):
    if osp.exists(colmap_pnts_f):
        colmap_pnts = np.load(colmap_pnts_f)
    else:
        assert colmap_dir is not None, "need all other params to create 3d points"
        manager = ColSceneManager(colmap_dir)
        rgb_K, rgb_D = manager.get_intrnxs()
        if calc_clear:
            clear_idxs = calc_clearness_score([manager.get_img_f(i+1) for i in range(len(manager))])[1]
            idx1, idx2 = clear_idxs[0] + 1, clear_idxs[0 + 6] + 1
        else:
            idx1, idx2 = 378, 29

        selector = ImagePointSelector([manager.get_img_f(idx1), manager.get_img_f(idx2)], save_dir=TMP_DIR)
        pnts = selector.select_points()
        colmap_pnts = triangulate_points(pnts, [manager.get_extrnxs(idx1), manager.get_extrnxs(idx2)], 
                                        {"intrinsics": rgb_K, "dist":rgb_D}, TMP_DIR)
        np.save(colmap_pnts_f, colmap_pnts)
    
    return colmap_pnts


def validate_in_colmap_space():
    colmap_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/calib_checker/calib_checker_recon"
    # eimg_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/ecam_code/raw_events/calib_checker/events_imgs"
    eimg_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/calib_checker/trig_eimgs"
    relcam_f = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/ecam_code/raw_events/calib_checker/rel_cam.json"
    save_dir = osp.join(TMP_DIR, "colmap_stereo_proj")

    manager = ColSceneManager(colmap_dir)
    os.makedirs(save_dir, exist_ok=True)
    rgb_K, rgb_D, ecam_K, ecam_D, R, T = load_relcam(relcam_f)

    #################################### FOR NON-CHECKER CHECKS ################################################
    # colmap_pnts_f = osp.join(TMP_DIR, "triangulated.npy")
    # if osp.exists(colmap_pnts_f):
    #     colmap_pnts = np.load(colmap_pnts_f)
    # else:
    #     idx1, idx2 = 1, 200
    #     # selector = ImagePointSelector([manager.get_img_f(idx1), manager.get_img_f(idx2)], save_dir=TMP_DIR)
    #     # pnts = selector.select_points()
    #     pnts = [find_chessboard_corners(manager.get_img(idx1)), find_chessboard_corners(manager.get_img(idx2))]
    #     colmap_pnts = triangulate_points(pnts, [manager.get_extrnxs(idx1), manager.get_extrnxs(idx2)], 
    #                                     {"intrinsics": rgb_K, "dist":rgb_D}, TMP_DIR)

    idx1, idx2 = 200, 1500
    pnts = [find_chessboard_corners(manager.get_img(idx1)), find_chessboard_corners(manager.get_img(idx2))]
    colmap_pnts = triangulate_points(pnts, [manager.get_extrnxs(idx1), manager.get_extrnxs(idx2)], 
                                    {"intrinsics": rgb_K, "dist":rgb_D}, TMP_DIR)

    grid_scale=0.39568848540866075/3
    colcams = [manager.get_extrnxs(i + 1) for i in range(len(manager))]
    ecams = map_cam({"R":R, "T":T}, colcams, grid_scale)

    def proj_fn(inp):
        img, extr = inp
        return proj_3d_pnts(img, ecam_K, extr, colmap_pnts, dist_coeffs=ecam_D)[1]

    eimgs = parallel_map(cv2.imread, sorted(glob.glob(osp.join(eimg_dir, "*.png"))))
    proj_eimgs = parallel_map(proj_fn, list(zip(eimgs, ecams)))

    
    def save_fn(inp):
        img, idx = inp
        cv2.imwrite(osp.join(save_dir, f"{str(idx).zfill(6)}.png"), img)
    
    parallel_map(save_fn, list(zip(proj_eimgs, list(range(len(eimgs))))), 
                 show_pbar=True, desc="saving projected")
    

def load_json_cam(cam_f):
    with open(cam_f, "r") as f:
        data = json.load(f)
        R, pos = np.array(data["orientation"]), np.array(data["position"])
        t = -(pos@R.T).T
        t = t.reshape(-1,1)
    
    return np.concatenate([R, t], axis=-1)


def load_json_intr(cam_f):
    with open(cam_f, "r") as f:
        data = json.load(f)
        fx = fy = data["focal_length"]
        cx, cy = data["principal_point"]
        k1, k2, k3 = data["radial_distortion"]
        p1, p2 = data["tangential_distortion"]
    
    return np.array([[fx, 0, cx],
                    [0,   fy, cy],
                    [0, 0, 1]]), (k1,k2,p1,p2)



def validate_ecamset():
    # scene = "grad_lounge_b2_v1"
    # scene = "atrium_b1_v1"
    scene = "halloween_b1_v1"
    objpnts_f = f"/scratch/matthew/projects/ecam-cam-datapipline/tmp/{scene}_triangulated.npy"

    ecamset = f"/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/{scene}/ecam_set"
    # ecamset = f"/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/{scene}/colcam_set"
    # ecamset = f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{scene}/trig_ecamset"

    colmap_dir = f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{scene}/{scene}_recon"

    save_dir_dicts = {"ecam_set":osp.join(TMP_DIR, f"{scene}_ecamset_proj"),
                      "colcam_set": osp.join(TMP_DIR, f"{scene}_colcamset_proj"),
                      "trig_ecamset":osp.join(TMP_DIR, f"{scene}_trig_ecamset_proj")}

    save_dir = save_dir_dicts[osp.basename(ecamset)]

    os.makedirs(save_dir, exist_ok=True)
    cam_fs = sorted(glob.glob(osp.join(ecamset, "camera", "*.json")))

    if "colcam_set" in ecamset:
        eimgs = sorted(glob.glob(osp.join(ecamset, "rgb", "1x", "*.png")))
    else:
        eimgs = np.load(osp.join(ecamset, "eimgs", "eimgs_1x.npy"), "r")

    ecam_K, ecam_D = load_json_intr(cam_fs[0])
    ecams = parallel_map(load_json_cam, cam_fs, show_pbar=True, desc="loading json cams") #[load_json_cam(f) for f in cam_fs]


    objpnts = load_objpnts(objpnts_f, colmap_dir, calc_clear=True)

    def proj_fn(inp):
        img, extr = inp
        return proj_3d_pnts(img, ecam_K, extr, objpnts, dist_coeffs=ecam_D)[1]


    if "colcam_set" in ecamset:
        eimgs = parallel_map(lambda x : cv2.imread(x), eimgs, show_pbar=True, desc="creating eimgs")
    else:
        eimgs = parallel_map(lambda x : np.stack([(x != 0).astype(np.uint8) * 255]*3, axis=-1), eimgs, show_pbar=True, desc="creating eimgs")
    proj_eimgs = parallel_map(proj_fn, list(zip(eimgs, ecams)), show_pbar=True, desc="projecting points")

    
    def save_fn(inp):
        img, idx = inp
        flag = cv2.imwrite(osp.join(save_dir, f"{str(idx).zfill(6)}.png"), img)
        assert flag, "save img failed"
    
    parallel_map(save_fn, list(zip(proj_eimgs, list(range(len(eimgs))))), 
                 show_pbar=True, desc="saving projected")

    save_f = f"{TMP_DIR}/{scene}_{osp.basename(ecamset)}.mp4"

    with contextlib.suppress(FileNotFoundError):
        os.remove(save_f)

    os.system(f"ffmpeg -framerate 60 -i {save_dir}/%06d.png -c:v libx264 -pix_fmt yuv420p -frames:v 5000 {save_f}")
    print("saved to", save_f)


if __name__ == "__main__":
    # validate_in_stereo_space()
    # validate_in_colmap_space()
    validate_ecamset()