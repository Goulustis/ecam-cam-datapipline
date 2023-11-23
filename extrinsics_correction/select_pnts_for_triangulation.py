import json
import os.path as osp
import glob
import matplotlib.pyplot as plt
from utils.images import calc_clearness_score
import numpy as np
import cv2

from extrinsics_correction.point_selector import ImagePointSelector
from extrinsics_visualization.colmap_scene_manager import ColSceneManager, proj_3d_pnts

WORK_DIR=osp.dirname(__file__)
SAVE_DIR=osp.join(WORK_DIR, "chosen_triang_pnts")
data_idxs = {"sofa_soccer_dragon": (1605, 1766)}

def select_triag_pnts(colmap_dir = None, output_dir=None):
    colmap_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/sofa_soccer_dragon_recon"
    output_dir = osp.join(SAVE_DIR, osp.basename(osp.dirname(colmap_dir))) if output_dir is None else output_dir

    manager = ColSceneManager(colmap_dir)

    idx1, idx2 = 1605, 1766

    # selector = ImagePointSelector([manager.get_img_f(idx) for idx in [idx1, idx2]], save=False, save_dir=output_dir)
    # selector.select_points()
    # selector.save_ref_img()

    extr1, extr2 = manager.get_extrnxs(idx1), manager.get_extrnxs(idx2)
    intrx, dist = manager.get_intrnxs()
    img_id1, img_id2 = manager.get_img_id(idx1), manager.get_img_id(idx2)

    np.save(osp.join(output_dir, f'{img_id1}_extrxs.npy'), extr1)
    np.save(osp.join(output_dir, f'{img_id2}_extrxs.npy'), extr2)
    np.save(osp.join(output_dir, "intrxs.npy"), {"intrinsics": intrx, "dist": dist})

    with open(osp.join(output_dir, "img_loc.txt"), "w") as f:
        f.write(manager.get_img_f(idx1) + "\n")
        f.write(manager.get_img_f(idx2) + "\n")

    # with open(osp.join(output_dir, "intrxs.json"), "w") as f:
    #     json.dump({"intrinsics": intrx.tolist(), "dist": dist.tolist()}, f, indent=4)

def find_correspondance(ori_pnts, selected_pnts):
    dists = np.sqrt(((ori_pnts[:, None] - selected_pnts[None])**2).sum(axis=-1))
    idxs = dists.argmin(axis=0)
    return idxs

def gen_point_img(pnts, radius=5):
    pnts = pnts / np.linalg.norm(pnts[1] - pnts[0])
    pix_gap = 64
    pnts = pnts * pix_gap
    pnts = pnts + pix_gap
    pnts = pnts.astype(int)

    w,h = pnts.max(axis=0) + pix_gap
    img_size = (h, w, 3)
    img = np.full(img_size, 255, dtype=np.uint8)

    for pnt in pnts:
        cv2.circle(img, pnt, radius, (0,0,0), -1)
    
    return img, pnts


def select_3d_coords():
    save_dir = "img_pnts/sofa_soccer_dragon"

    # Initialize 3D points
    objp = np.zeros((5*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * 3  # Scale and switch X and Y positions

    # Dropping the last coordinate (Z-coordinate) for 2D plotting
    objp_2d = objp[:, :2]
    
    proj_img, norm_pnts = gen_point_img(objp_2d)

    # Save the plot as an image
    pnt_plot_f = osp.join(WORK_DIR, '2d_points_plot.png')
    cv2.imwrite(pnt_plot_f, proj_img)

    selector = ImagePointSelector([pnt_plot_f], save=False)
    pnts = np.array(selector.select_points()[0])
    pnt_scale = np.linalg.norm(pnts[1] - pnts[0])
    ori_scale = np.linalg.norm(norm_pnts[1] - norm_pnts[0])

    pnts = pnts/pnt_scale
    norm_objp = norm_pnts/ori_scale
    corr_idxs = find_correspondance(norm_objp, pnts)

    chosen_pnts = objp[corr_idxs]

    np.save(osp.join(save_dir, "corr_3d_pnts.pny"), chosen_pnts)

def load_output_dir(path):
    pnts_fs = sorted(glob.glob(osp.join(path, "*_pnts.npy")))
    extrxs_fs = sorted(glob.glob(osp.join(path, "*_extrxs.npy")))
    intrxs_f = osp.join(path, "intrxs.npy")

    
    return {"pnts": [np.load(pnts_f) for pnts_f in pnts_fs],
            "extr": [np.load(extrxs_f) for extrxs_f in extrxs_fs],
            "intr": np.load(intrxs_f, allow_pickle=True).item()}


def execute_triangulate_points(points1, points2, intrinsics1, dist_coeffs1, intrinsics2, dist_coeffs2, extrinsics1, extrinsics2):
    """
    Triangulate 3D points from two sets of corresponding 2D points.

    :param points1: List of (x, y) tuples of points in the first image.
    :param points2: List of (x, y) tuples of points in the second image.
    :param intrinsics1: Intrinsic matrix of the first camera.
    :param dist_coeffs1: Distortion coefficients of the first camera.
    :param intrinsics2: Intrinsic matrix of the second camera.
    :param dist_coeffs2: Distortion coefficients of the second camera.
    :param extrinsics1: 4x4 extrinsic matrix [R | t] of the first camera.
    :param extrinsics2: 4x4 extrinsic matrix [R | t] of the second camera.
    :return: Array of 3D points.
    """
    # Extract R and t from the extrinsic matrix
    R1, t1 = extrinsics1[:3, :3], extrinsics1[:3, 3:]
    R2, t2 = extrinsics2[:3, :3], extrinsics2[:3, 3:]

    # Undistort points
    points1_undistorted = cv2.undistortPoints(np.array(points1, dtype=np.float32), intrinsics1, dist_coeffs1)
    points2_undistorted = cv2.undistortPoints(np.array(points2, dtype=np.float32), intrinsics2, dist_coeffs2)

    # Triangulate points
    points_3d_hom = cv2.triangulatePoints(np.hstack((R1, t1)), np.hstack((R2, t2)), points1_undistorted, points2_undistorted)
    # Convert from homogeneous to 3D coordinates
    points_3d = cv2.convertPointsFromHomogeneous(points_3d_hom.T)

    return points_3d

def calculate_reprojection_error(points_3d, points2d_1, points2d_2, intrinsics1, dist_coeffs1, intrinsics2, dist_coeffs2, extrinsics1, extrinsics2):
    """
    Calculate the reprojection error of triangulated 3D points.

    :param points_3d: Triangulated 3D points.
    :param points2d_1: Original 2D points in the first image.
    :param points2d_2: Original 2D points in the second image.
    :param intrinsics1: Intrinsic matrix of the first camera.
    :param dist_coeffs1: Distortion coefficients of the first camera.
    :param intrinsics2: Intrinsic matrix of the second camera.
    :param dist_coeffs2: Distortion coefficients of the second camera.
    :param extrinsics1: 4x4 extrinsic matrix [R | t] of the first camera.
    :param extrinsics2: 4x4 extrinsic matrix [R | t] of the second camera.
    :return: Mean reprojection error.
    """
    # Extract R and t from extrinsic matrices
    # extrinsics1 = np.linalg.inv(extrinsics1)
    # extrinsics2 = np.linalg.inv(extrinsics2)
    R1, t1 = extrinsics1[:3, :3], extrinsics1[:3, 3:]
    R2, t2 = extrinsics2[:3, :3], extrinsics2[:3, 3:]

    # Reproject points back to 2D for both cameras
    points2d_reproj_1, _ = cv2.projectPoints(points_3d, R1, t1, intrinsics1, dist_coeffs1)
    points2d_reproj_2, _ = cv2.projectPoints(points_3d, R2, t2, intrinsics2, dist_coeffs2)

    # Calculate errors
    error_1 = np.sqrt(np.sum((points2d_1 - points2d_reproj_1.squeeze())**2, axis=1))
    error_2 = np.sqrt(np.sum((points2d_2 - points2d_reproj_2.squeeze())**2, axis=1))

    # Compute mean reprojection error
    mean_error_1 = np.mean(error_1)
    mean_error_2 = np.mean(error_2)
    total_mean_error = (mean_error_1 + mean_error_2) / 2

    return total_mean_error


def triangulate_points(pnts, extr, intr):
    pnts1, pnts2 = pnts
    extrx1, extrx2 = extr
    intrinsics, dist = intr["intrinsics"], intr["dist"]
    pnts_3d = execute_triangulate_points(pnts1, pnts2, intrinsics, dist, intrinsics, dist, extrx1, extrx2)

    #### sanity check
    # err = calculate_reprojection_error(pnts_3d, pnts1, pnts2, intrinsics, dist, intrinsics, dist, extrx1, extrx2)
    # img1 = cv2.imread("/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/sofa_soccer_dragon_recon/images/01604.png")
    # img2 = cv2.imread("/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/sofa_soccer_dragon_recon/images/01765.png")
    # prj_img1 = proj_3d_pnts(img1, intrinsics, extrx1, pnts_3d, dist_coeffs=dist)[-1]
    # prj_img2 = proj_3d_pnts(img2, intrinsics, extrx2, pnts_3d, dist_coeffs=dist)[-1]
    # assert 0
    return pnts_3d



# def main():
#     colmap_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/sofa_soccer_dragon_recon"
#     output_dir = osp.join(SAVE_DIR, osp.basename(osp.dirname(colmap_dir)))
#     select_triag_pnts(colmap_dir, output_dir)
#     triangulate_points(*load_output_dir(output_dir))
    
    



if __name__ == "__main__":
    colmap_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/sofa_soccer_dragon_recon"
    output_dir = osp.join(SAVE_DIR, osp.basename(osp.dirname(colmap_dir)))
    # select_triag_pnts(colmap_dir, output_dir)
    triangulate_points(**load_output_dir(output_dir))
    # select_3d_coords()

