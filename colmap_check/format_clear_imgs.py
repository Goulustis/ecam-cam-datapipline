import os
import os.path as osp
from utils.images import calc_clearness_score
from utils.misc import parallel_map
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(32)


def copy_img(img_fs, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    print(f"saving {len(img_fs)} imgs to:", dst_dir)
    def copy_fn(img_f):
        dst_f = osp.join(dst_dir, osp.basename(img_f))
        shutil.copyfile(img_f, dst_f)
    
    parallel_map(copy_fn, img_fs, show_pbar=True, desc="saving imgs")


def main():
    """
    create:
    <scene>_checker_recons/
        images/
        trig_eimgs/
    
    images: images used for colmap
    trig_eimgs: corresponding event images, use third_party/e2img to create
    """
    col_img_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/black_seoul_b3_v3/black_seoul_b3_v3_recon/images"
    trig_eimg_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/black_seoul_b3_v3/trig_eimgs"
    
    workdir = osp.dirname(trig_eimg_dir)
    dest_dir = osp.join(workdir, f"{osp.basename(workdir)}_check_recons")
    os.makedirs(dest_dir, exist_ok=True)

    col_img_fs = np.array(sorted(glob.glob(osp.join(col_img_dir, "*.png"))))
    trig_eimg_fs = np.array(sorted(glob.glob(osp.join(trig_eimg_dir, "*.png"))))

    # req_idxs = sorted([0, 25, 747, 774, 760, 931, 1508, 1707, 1724])
    req_idxs = sorted([0,1,2,24, 25, 26, 746,747,748, 773,774,775,759, 760, 761, 930,931,932, 1507,1508,1509, 1706, 1707, 1708,  1723, 1724, 1725])
    req_col_img_fs = col_img_fs[req_idxs]
    req_eimg_fs = trig_eimg_fs[req_idxs]

    clear_fs, idxs, scores = calc_clearness_score(col_img_fs)

    chosen_idxs = np.random.choice(idxs[:33], 32)
    chosen_col_fs = sorted(list(set(col_img_fs[chosen_idxs].tolist() + req_col_img_fs.tolist())))
    chosen_eimg_fs = sorted(list(set(trig_eimg_fs[chosen_idxs].tolist() + req_eimg_fs.tolist())))

    copy_img(chosen_col_fs, osp.join(dest_dir, "images"))
    copy_img(chosen_eimg_fs, osp.join(dest_dir, "trig_eimgs"))



if __name__ == "__main__":
    main()