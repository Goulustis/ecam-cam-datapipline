import os.path as osp
import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.linalg import svd
from scipy import optimize
from tqdm import tqdm
import argparse

from extrinsics_correction.point_selector import ImagePointSelector
from extrinsics_visualization.colmap_scene_manager import ColmapSceneManager, proj_3d_pnts, draw_2d_pnts
from utils.images import calc_clearness_score
from utils.misc import parallel_map

WORK_DIR=osp.dirname(__file__)
SAVE_DIR=osp.join(WORK_DIR, "chosen_triang_pnts")
data_idxs = {"sofa_soccer_dragon": (1605, 1766)}

def detect_chessboard(img):
    ret, pnts = cv2.findChessboardCorners(img, (5, 8), None)
    assert ret
    return pnts

def select_triag_pnts(colmap_dir = None, output_dir=None, use_score=False, use_checker=False):
    output_dir = osp.join(SAVE_DIR, osp.basename(osp.dirname(colmap_dir))) if output_dir is None else output_dir

    manager = ColmapSceneManager(colmap_dir)

    if use_score:
        clear_idxs = calc_clearness_score([manager.get_img_f(i+1) for i in range(len(manager))])[1]
        idx1, idx2 = clear_idxs[1] + 1, clear_idxs[3] + 1
    else:
        idx1, idx2 = 375, 409
        # idx1, idx2 = 3, 54
    
    print("img used:", osp.basename(manager.get_img_f(idx1)), osp.basename(manager.get_img_f(idx2)))

    selector = ImagePointSelector([manager.get_img_f(idx) for idx in [idx1, idx2]], save=True, save_dir=output_dir)
    selector.select_points()
    selector.save_ref_img()

    extr1, extr2 = manager.get_extrnxs(idx1), manager.get_extrnxs(idx2)
    intrx, dist = manager.get_intrnxs()
    img_id1, img_id2 = manager.get_img_id(idx1), manager.get_img_id(idx2)

    np.save(osp.join(output_dir, f'{img_id1}_extrxs.npy'), extr1)
    np.save(osp.join(output_dir, f'{img_id2}_extrxs.npy'), extr2)
    np.save(osp.join(output_dir, "intrxs.npy"), {"intrinsics": intrx, "dist": dist})

    with open(osp.join(output_dir, "img_loc.txt"), "w") as f:
        f.write(manager.get_img_f(idx1) + "\n")
        f.write(manager.get_img_f(idx2) + "\n")

    return int(img_id1), int(img_id2)

def find_correspondance(ori_pnts, selected_pnts):
    dists = np.sqrt(((ori_pnts[:, None] - selected_pnts[None])**2).sum(axis=-1))
    idxs = dists.argmin(axis=0)
    return idxs

def gen_point_img(pnts, radius=5):
    #NOTE: THESE ARE INNER CORNORS
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


def select_3d_checker_coords(out_dir, use_checker=False):

    # Initialize 3D points
    objp = np.zeros((5*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * 3  # Scale and switch X and Y positions

    # Dropping the last coordinate (Z-coordinate) for 2D plotting
    objp_2d = objp[:, :2]
    
    proj_img, norm_pnts = gen_point_img(objp_2d)

    # Save the plot as an image
    pnt_plot_f = osp.join(WORK_DIR, '2d_points_plot.png')
    cv2.imwrite(pnt_plot_f, proj_img)

    if not use_checker:
        selector = ImagePointSelector([pnt_plot_f], save=False)
        pnts = np.array(selector.select_points()[0])
        pnt_scale = np.linalg.norm(pnts[1] - pnts[0])
        ori_scale = np.linalg.norm(norm_pnts[1] - norm_pnts[0])

        pnts = pnts/pnt_scale
        norm_objp = norm_pnts/ori_scale
        corr_idxs = find_correspondance(norm_objp, pnts)

        chosen_pnts = objp[corr_idxs]
    else:
        chosen_pnts = objp

    np.save(osp.join(out_dir, "corres_3d.npy"), chosen_pnts)

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


def triangulate_points(pnts, extr, intr, output_dir = None):
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
    if output_dir is not None:
        np.save(osp.join(output_dir, "triangulated.npy"), pnts_3d.squeeze())

    return pnts_3d




def find_rigid_transform(A, B):
    """
    Find the scale, rotation, and translation to map points in A to points in B.
    
    :param A: Nx3 numpy array of 3D points.
    :param B: Nx3 numpy array of 3D points.
    :return: scale, rotation matrix, translation vector
    """
    assert A.shape == B.shape, "The point sets must have the same shape."

    # Center the points (translation)
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute scale
    scale = np.sqrt(np.sum(B_centered**2) / np.sum(A_centered**2))

    # Scale A
    A_scaled = A_centered * scale

    # Compute rotation
    H = A_scaled.T @ B_centered
    U, _, Vt = svd(H)
    rotation = Vt.T @ U.T

    # Ensure a right-handed coordinate system
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    # Compute final translation
    translation = centroid_B - scale * (rotation @ centroid_A)

    return scale, rotation, translation


def find_scale(out_dir):
    triang_f = osp.join(out_dir, "triangulated.npy")
    corres_f = osp.join(out_dir, "corres_3d.npy")
    triang, corres = np.load(triang_f), np.load(corres_f)

    # find s,R,t maps corres --> triang
    s, R, t = find_rigid_transform(corres, triang)
    
    error = np.sqrt((s*(corres@R.T) + t - triang)**2).mean()
    print("error:", error)
    print("scale:", s)
    return s


def pnp_extrns(objpnts, pnts_2d, intrxs, dist, ini_R=None, ini_tvec=None):
    pnts_2d = pnts_2d.squeeze()

    if ini_R is None:
        success, rvec, tvec = cv2.solvePnP(objpnts, pnts_2d, intrxs, dist)
    else:
        initial_rvec, _ = cv2.Rodrigues(ini_R)
        # success, rvec, tvec, inliers = cv2.solvePnPRansac(objpnts, pnts_2d, intrxs, dist, flags=cv2.SOLVEPNP_ITERATIVE, rvec=initial_rvec, tvec=ini_tvec)
        success, rvec, tvec = cv2.solvePnP(
                        objpnts, 
                        pnts_2d, 
                        intrxs, 
                        dist, 
                        rvec=initial_rvec, 
                        tvec=ini_tvec, 
                        useExtrinsicGuess=True, 
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
    R, _ = cv2.Rodrigues(rvec)

    assert success
    return R, tvec



def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



def plot_3d_points(points):
    """
    Plot a list of 3D points using Matplotlib.

    Args:
        points (list of tuple): List of 3D points as (x, y, z) tuples.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z coordinates from the points
    x_points, y_points, z_points = zip(*points)

    # Plot the 3D points
    ax.scatter(x_points, y_points, z_points, c='b', marker='o', label='3D Points')

    # Set axis labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    set_axes_equal(ax)

    # Set the title
    plt.title('3D Point Cloud')

    # Show the plot
    plt.legend()
    plt.show()

def plot_3d_points_2set(points1, points2):
    """
    Plot two sets of 3D points using Matplotlib in different colors.

    Args:
        points1 (list of tuple): List of 3D points as (x, y, z) tuples for the first set.
        points2 (list of tuple): List of 3D points as (x, y, z) tuples for the second set.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z coordinates from the points for each set
    x_points1, y_points1, z_points1 = zip(*points1)
    x_points2, y_points2, z_points2 = zip(*points2)

    # Plot the first set of 3D points in blue
    ax.scatter(x_points1, y_points1, z_points1, c='b', marker='o', label='Set 1')

    # Plot the second set of 3D points in red
    ax.scatter(x_points2, y_points2, z_points2, c='r', marker='^', label='Set 2')

    # Set axis labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    set_axes_equal(ax)

    # Set the title
    plt.title('3D Point Cloud')

    # Show the legend to differentiate the sets
    plt.legend()
    plt.show()


def pnp_find_rel_cam(out_dir, work_dir):
    triag_f = osp.join(out_dir, "triangulated.npy")
    idx1, idx2 = (0, 40)

    eimg_fs = sorted(glob.glob(osp.join(work_dir, "trig_eimgs", "*.png")))
    rel_cam_f = osp.join(work_dir, "rel_cam.json")

    objpnts = np.load(triag_f)
    # idx1, idx2 = idx1 - 1, idx2 - 1
    idx1, idx2 = idx1, idx2
    eimg_f1, eimg_f2 = eimg_fs[idx1], eimg_fs[idx2]
    with open(rel_cam_f, "r") as f:
        data = json.load(f)
        ecam_intrx, ecam_dist = np.array(data["M2"]), np.array(data["dist2"])
        fr, ft = np.array(data["R"]), np.array(data["T"])
    
    make_chosen_f = lambda x : osp.join(out_dir, osp.basename(x).split(".")[0] + "_eimg.npy")
    f1_chosen_f, f2_chosen_f = make_chosen_f(eimg_f1), make_chosen_f(eimg_f2)
    
    if osp.exists(f1_chosen_f):
        pnts1, pnts2 = np.load(f1_chosen_f), np.load(f2_chosen_f)
    else:
        selector = ImagePointSelector([eimg_f1, eimg_f2], save_fs = [f1_chosen_f, f2_chosen_f])
        pnts1, pnts2 = selector.select_points()
    pnts1, pnts2 = pnts1.astype(np.float32), pnts2.astype(np.float32)
    rgb_cam_fs = sorted(glob.glob(osp.join(out_dir, "*_extrxs.npy")))
    Rs, Ts = [], []
    ecam_Rs, ecam_Ts = [], []
    manager = ColmapSceneManager(osp.join(work_dir, "sofa_soccer_dragon_recon"))
    for i, (pnt, rgb_cam_f)  in enumerate(zip([pnts1, pnts2], rgb_cam_fs)):
        rgb_cam = np.load(rgb_cam_f)
        rgb_cam = manager.get_extrnxs([idx1, idx2][i] + 1)
        rgb_R, rgb_t = rgb_cam[:3,:3], rgb_cam[:3,3:]
        # ecam_R, ecam_t = pnp_extrns(objpnts, pnt, ecam_intrx, ecam_dist, rgb_R, rgb_t)
        ecam_R, ecam_t = pnp_extrns(objpnts, pnt, ecam_intrx, ecam_dist)
        
        R_rel = ecam_R@rgb_R.T
        T_rel = rgb_R.T@(ecam_t - rgb_t)
        Rs.append(R_rel), Ts.append(T_rel)
        ecam_Rs.append(ecam_R), ecam_Ts.append(ecam_t)
    
    assert 0


def scale_opt(output_dir, work_dir, colmap_dir, rgb_ids):
    relcam_f = osp.join(work_dir, "rel_cam.json")
    with open(relcam_f, "r") as f:
        data = json.load(f)
        ecam_K, ecam_D, R, T = [np.array(data[e]) for e in ["M2", "dist2", "R", "T"]]
    
    objpoints = np.load(osp.join(output_dir, "triangulated.npy"))

    idx1, idx2 = rgb_ids
    eimg_fs = sorted(glob.glob(osp.join(work_dir, "trig_eimgs", "*.png")))
    selector = ImagePointSelector([eimg_fs[idx1], eimg_fs[idx2]], save_dir=output_dir, end_fix="opt")

    pnts_2d_fs = sorted(glob.glob(osp.join(output_dir, "*_opt.npy")))
    if len(pnts_2d_fs) == 0:
        pnts_2d = selector.select_points()
    else:
        pnts_2d = [np.load(f) for f in pnts_2d_fs]
    
    manager = ColmapSceneManager(colmap_dir)
    rgb_extrs = [manager.get_extrnxs(idx1+1), manager.get_extrnxs(idx2+1)]

    def loss_fn(scale):
        loss = 0
        for i, (rgb_extr, pnt_2d) in enumerate(zip(rgb_extrs, pnts_2d)):
            rgb_R, rgb_T = rgb_extr[:3,:3], rgb_extr[:3,3:]
            ecam_R, ecam_T = R@rgb_R, R@rgb_T + T*scale
            proj_pnts = proj_3d_pnts(None, ecam_K, np.concatenate([ecam_R, ecam_T], axis=1), objpoints, dist_coeffs=ecam_D)[0].squeeze()
            loss = loss + ((proj_pnts - pnt_2d)**2).mean()
        
        return loss
    
    print("initial loss:", loss_fn(0.158))
    res = optimize.minimize(loss_fn, (0.158, ), method='Powell', bounds=[(0, None)])
    scale_solv = res.x[0]
    print("final loss:", loss_fn(scale_solv))


    print("optim res:", res)
    print("optim scale:", scale_solv)
    with open(osp.join(colmap_dir, "colmap_scale.txt"), "w") as f:
        f.write(str(scale_solv))

    ### sanity checking ###
    ecam_Rs, ecam_Ts = [], []
    for i, (rgb_extr, pnt_2d) in enumerate(zip(rgb_extrs, pnts_2d)):
        rgb_R, rgb_T = rgb_extr[:3,:3], rgb_extr[:3,3:]
        
        ecam_R, ecam_T = R@rgb_R, R@rgb_T + T*scale_solv
        ecam_Rs.append(ecam_R), ecam_Ts.append(ecam_T)
    
    ### project points for sanity checking
    for i, eimg_f in enumerate([eimg_fs[idx1], eimg_fs[idx2]]):
        eimg = cv2.imread(eimg_f)
        _, proj_eimg = proj_3d_pnts(eimg, ecam_K, np.concatenate([ecam_Rs[i], ecam_Ts[i]], axis=1), objpoints, dist_coeffs=ecam_D)
        cv2.imwrite(osp.join(output_dir, osp.basename(eimg_f)), proj_eimg)
    
    for i, img_idx in enumerate([idx1+1, idx2+1]):
        img_f = manager.get_img_f(img_idx)
        img = cv2.imread(img_f)
        rgb_K, rgb_D = manager.get_intrnxs()
        drawn_img = proj_3d_pnts(img, rgb_K, manager.get_extrnxs(img_idx), objpoints, dist_coeffs=rgb_D)[1]
        cv2.imwrite(osp.join(output_dir, f"rgb_{osp.basename(img_f)}"), drawn_img)
    
    img_f = manager.get_img_f(200)
    img = cv2.imread(img_f)
    rgb_K, rgb_D = manager.get_intrnxs()
    drawn_img = proj_3d_pnts(img, rgb_K, manager.get_extrnxs(200), objpoints, dist_coeffs=rgb_D)[1]
    cv2.imwrite(osp.join(output_dir, f"rgb_extra_{osp.basename(img_f)}"), drawn_img)

def save_blur_imgs(colmap_dir):
    import shutil

    manager = ColmapSceneManager(colmap_dir)
    img_fs = [manager.get_img_f(i+1) for i in range(len(manager))]
    fs, idxs, scores = calc_clearness_score(img_fs)
    
    scores = scores[idxs]
    fs = np.array(fs)[scores < 56]

    save_dir = "tmp/blur_imgs"
    os.makedirs(save_dir, exist_ok=True)
    for i, img_f in enumerate(tqdm(fs, desc="saving blur fs")):
        save_f = osp.join(save_dir, f"{str(i).zfill(4)}.png")
        shutil.copy(img_f, save_f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="oc_v11_t1")
    args = parser.parse_args()
    scene = args.scene
    

    colmap_dir = f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{scene}/{scene}_recon"
    work_dir = f"/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/{scene}"

    # output_dir = osp.join(SAVE_DIR, osp.basename(osp.dirname(colmap_dir)))
    output_dir = osp.join(SAVE_DIR, "_".join(osp.basename(colmap_dir).split("_")[:-1]))
    use_checker=False
    rgb_ids = select_triag_pnts(colmap_dir, output_dir, use_score=False, use_checker=use_checker)
    triangulate_points(**load_output_dir(output_dir), output_dir=output_dir)
    # select_3d_checker_coords(output_dir, use_checker=use_checker)
    # find_scale(output_dir)
    
    scale_opt(output_dir, work_dir, colmap_dir, rgb_ids=rgb_ids)
