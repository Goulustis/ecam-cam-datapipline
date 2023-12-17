import glob
import os
import os.path as osp
import cv2
from functools import partial

from extrinsics_visualization.colmap_scene_manager import ColSceneManager, proj_3d_pnts
from stereo_calib.validate_stereo import load_relcam, load_objpnts
from extrinsics_creator.create_rel_cam import map_cam
from utils.misc import parallel_map
from viz.comb_vid import concatenate_videos

OUT_DIR=osp.join(osp.dirname(__file__), "proj_dir")


def proj_fn(inp, objpnts, K, D):
        img, extr = inp
        return proj_3d_pnts(img, K, extr, objpnts, dist_coeffs=D)[1]


def save_imgs(imgs, save_dir, img_names=None):
    os.makedirs(save_dir, exist_ok=True)
    if img_names is None:
        img_names = [f"{str(i).zfill(6)}" for i in range(len(imgs))]

    def save_fn(inp):
        img, name = inp
        save_f = osp.join(save_dir, name)
        cv2.imwrite(save_f, img)
    
    parallel_map(save_fn, list(zip(imgs, img_names)), show_pbar=True, desc="saving imgs")
    print("saved all to:", save_dir)  ## for ease of viewing
         
    


def main():
    """
    colmap_scene_f:
        <scene>_recons/
            images/
            trig_eimgs/
            sparse/   <--- results
    scale: scale of relative R,T to colmap world scale, use manuel_scale_find.py to find it.
    find points in 3d and project them to rgb and event camera.
    """
    colmap_scene_f = "/scratch-ssd/workdir/black_seoul_b3_v3_check_recons"
    rel_cam_f = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/ecam_code/raw_events/calib_checker/rel_cam.json"
    # scale = 0.12870096 ## b1
    scale = 0.12634888 ## b3

    dst_dir = osp.join(OUT_DIR, osp.basename(colmap_scene_f)[:-7])
    obj_f = osp.join(dst_dir, "triangulated.npy")

    os.makedirs(dst_dir, exist_ok=True)
    objpnts = load_objpnts(obj_f, colmap_scene_f, calc_clear=True, use_checker=True)


    manager = ColSceneManager(colmap_scene_f)
    trig_eimg_fs = sorted(glob.glob(osp.join(colmap_scene_f, "trig_eimgs", "*.png")))
    _, _, ev_K, ev_D, R, T = load_relcam(rel_cam_f)

    colcams = manager.get_all_extrnxs()
    ecams = map_cam({"R":R, "T":T}, colcams, scale)

    ecam_prj_fn = partial(proj_fn, objpnts=objpnts, K=ev_K, D=ev_D)
    eimgs = parallel_map(lambda f: cv2.imread(f), trig_eimg_fs)
    prj_eimgs = parallel_map(ecam_prj_fn, list(zip(eimgs, ecams)), show_pbar=True, desc="projecting eimgs")

    col_K, col_D = manager.get_intrnxs()
    colcam_prj_fn = partial(proj_fn, objpnts=objpnts, K=col_K, D=col_D)
    prj_imgs = parallel_map(colcam_prj_fn, list(zip(manager.get_all_imgs(), colcams)), show_pbar=True, desc="projecting imgs")

    comb_prjs = concatenate_videos(prj_eimgs, prj_imgs)
    save_imgs(comb_prjs, osp.join(dst_dir), [osp.basename(f) for f in manager.img_fs])


if __name__ == "__main__":
    main()