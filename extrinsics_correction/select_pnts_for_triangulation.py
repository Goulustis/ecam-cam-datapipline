from extrinsics_visualization.colmap_scene_manager import ColSceneManager
import os
import os.path as osp
import glob
import matplotlib.pyplot as plt
from utils.images import calc_clearness_score
from extrinsics_correction.point_selector import ImagePointSelector
import numpy as np

WORK_DIR=osp.dirname(__file__)
SAVE_DIR=osp.join(WORK_DIR, "chosen_triang_pnts")
data_idxs = {"sofa_soccer_dragon": (1605, 1766)}

def select_triag_pnts():
    colmap_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/sofa_soccer_dragon_recon"
    manager = ColSceneManager(colmap_dir)

    idx1, idx2 = 1605, 1766

    selector = ImagePointSelector([manager.get_img_f(idx) for idx in [idx1, idx2]], save=True, save_dir=osp.join(SAVE_DIR, osp.basename(osp.dirname(colmap_dir))))
    selector.select_points()
    selector.save_ref_img()




def select_3d_coords():

    # Initialize 3D points
    objp = np.zeros((5*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * 3  # Scale and switch X and Y positions

    # Dropping the last coordinate (Z-coordinate) for 2D plotting
    objp_2d = objp[:, :2]

    # Create a 2D plot
    plt.figure(figsize=(10, 8))
    for idx, point in enumerate(objp_2d):
        plt.scatter(point[0], point[1], c='blue', marker='o')
        plt.text(point[0], point[1], str(idx), fontsize=9, ha='right')

    # Set labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Plot of 3D Points (Z-coordinate dropped)')

    # Save the plot as an image
    pnt_plot_f = osp.join(WORK_DIR, '2d_points_plot.png')
    plt.savefig(pnt_plot_f, dpi=300)
    plt.close()

    selector = ImagePointSelector([pnt_plot_f], save=False)

    # SELECT POINTS, FIND COORESPONDENCE, SAVE THE 3D POINTS





if __name__ == "__main__":
    select_triag_pnts()