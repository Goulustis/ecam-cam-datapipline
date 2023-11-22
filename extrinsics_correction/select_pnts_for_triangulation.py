import json
import os.path as osp
import glob
import matplotlib.pyplot as plt
from utils.images import calc_clearness_score
import numpy as np
import cv2

from extrinsics_correction.point_selector import ImagePointSelector
from extrinsics_visualization.colmap_scene_manager import ColSceneManager

WORK_DIR=osp.dirname(__file__)
SAVE_DIR=osp.join(WORK_DIR, "chosen_triang_pnts")
data_idxs = {"sofa_soccer_dragon": (1605, 1766)}

def select_triag_pnts():
    colmap_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/sofa_soccer_dragon_recon"

    output_dir = osp.join(SAVE_DIR, osp.basename(osp.dirname(colmap_dir)))

    manager = ColSceneManager(colmap_dir)

    idx1, idx2 = 1605, 1766

    selector = ImagePointSelector([manager.get_img_f(idx) for idx in [idx1, idx2]], save=True, save_dir=output_dir)
    selector.select_points()
    selector.save_ref_img()

    extr1, extr2 = manager.get_extrnxs(idx1), manager.get_extrnxs(idx2)
    intrx, dist = manager.get_intrnxs()
    img_id1, img_id2 = manager.get_img_id(idx1), manager.get_img_id(idx2)

    np.save(osp.join(output_dir, f'{img_id1}_extrxs.npy'), extr1)
    np.save(osp.join(output_dir, f'{img_id2}_extrxs.npy'), extr2)
    
    with open(osp.join(output_dir, "intrxs.json"), "w") as f:
        json.dump({"intrinsics": intrx.tolist(), "dist": dist.tolist()}, f, indent=4)

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





if __name__ == "__main__":
    select_triag_pnts()
    # select_3d_coords()
