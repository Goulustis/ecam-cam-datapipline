import numpy as np
import os
import os.path as osp
import glob
import cv2
from tqdm import tqdm
import argparse

import colmap_find_scale.read_write_model as colmap_utils
from extrinsics_visualization.colmap_scene_manager import ColSceneManager
from extrinsics_visualization.ev_scene_manager import EvSceneManager
from extrinsics_visualization.colmap_scene_manager import proj_3d_pnts, concat_imgs

def create_video_from_images(imgs, output_filename='output.avi', fps=15):
    """
    Create a video file from a list of images.

    Args:
    imgs (list of np.array): List of images, each as a NumPy array.
    output_filename (str): Name of the output video file.
    fps (int): Frames per second for the output video.

    Returns:
    None
    """

    if not imgs:
        raise ValueError("The list of images is empty.")

    # Determine the size of the images
    height, width, layers = imgs[0].shape
    size = (width, height)

    # Initialize the VideoWriter object
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for img in tqdm(imgs, desc="saving vid"):
        if img.shape != (height, width, layers):
            raise ValueError("All images must be of the same size and number of channels.")
        out.write(img)

    out.release()


def main(colmap_dir, ev_recon_img_dir, ecam_f, rel_cam_f):
    # colmap_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/atrium_b2_v1_recons"
    # ev_recon_img_dir="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1/ev_imgs/e2calib"
    # ecam_f = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1/e_cams.npy"
    # rel_cam_f = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1/rel_cam.json"

    colmap_manager = ColSceneManager(colmap_dir)
    ev_manager = EvSceneManager(ev_recon_img_dir, ecam_f, rel_cam_f)

    colmap_manager.view_img_points(img_idx=3)  # initialize some points for projection
    pnts_idxs, pnts_3d = colmap_manager.chosen_points["idxs"],  colmap_manager.chosen_points["xyzs"]

    vid_frames = []
    for i in tqdm(range(len(ev_manager)), desc="projecting images"):
        ev_img, ev_intrxs, ev_extrxs, ev_dist_coeffs = ev_manager.get_data(i)
        
        rgb_prj_img = colmap_manager.view_img_points(i+1,chosen_pnt_idxs=pnts_idxs)
        ev_prj_img = proj_3d_pnts(ev_img, ev_intrxs, ev_extrxs, pnts_idxs, pnts_3d, ev_dist_coeffs)[1]
        comb_img = concat_imgs(rgb_prj_img, ev_prj_img)
        vid_frames.append(comb_img)
    
    save_f = osp.join(osp.dirname(ecam_f), "ecam_extr.mp4")
    print("saving to", save_f)
    create_video_from_images(vid_frames, save_f)
    print("saving complete")
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_dir", help="folder to xxx_recons")
    parser.add_argument("--ev_imgs_dir", help="folder containing *.png")
    parser.add_argument("--ecam_f", help="folder containing the extrinsics of the event camera")
    parser.add_argument("--relcam_f", help="path to rel_cam.json")
    args = parser.parse_args()


    main(args.colmap_dir, args.ev_imgs_dir, args.ecam_f, args.relcam_f)

    
