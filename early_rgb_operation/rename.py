import os
import os.path as osp
import shutil
from tqdm import tqdm
import glob

def main():
    src_dir = "timer_rgb_checker"
    targ_dir = "timer_rgb_checker_png"

    img_fs = sorted(glob.glob(osp.join(src_dir, "*.png")))

    os.makedirs(targ_dir, exist_ok=True)
    
    for i, img_f in tqdm(enumerate(img_fs),total=len(img_fs)):
        dest = osp.join(targ_dir, str(i).zfill(6) + ".png")
        shutil.copy(img_f, dest)


if __name__ == "__main__":
    main()
