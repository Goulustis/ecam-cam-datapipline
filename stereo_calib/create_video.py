from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np
import cv2
import glob
import os.path as osp
from tqdm import tqdm
from utils.misc import parallel_map

"""
create a video for corresponding videos; sanity check to make sure things are not broken
"""

def upscale(img, targ_h):
    ## asssume grayscale
    if len(img.shape) !=3 :
        img = img[..., None]

    h, w, c = img.shape
    new_img = np.zeros((targ_h, w, c), dtype=img.dtype)
    st_h = (targ_h - h)//2

    new_img[st_h:st_h + h, 0:w] = img
    return new_img


def img_concat(imgs):
    ## assume only 2 list
    hs = np.array([img.shape[0] for img in imgs])
    max_h = max(hs)

    l_img = imgs[hs.argmax()]
    s_img = imgs[hs.argmin()]

    s_img = upscale(s_img, max_h)
    return np.concatenate([l_img, s_img], axis=1)



def main():
    e_img_fs = sorted(glob.glob(osp.join("/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/lab_c1/trig_eimgs", "*")))
    c_img_fs = sorted(glob.glob(osp.join("/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/rgb-evs-cam-drivers/data_rgb/lab_c1_recons/images", "*")))
    n_frames = min(len(e_img_fs), len(c_img_fs))

    e_img_fs = e_img_fs[:n_frames]
    c_img_fs = c_img_fs[:n_frames]

    def concat_fn(inp):
        i, e_f, c_f = inp
        e_img = cv2.imread(e_f, cv2.IMREAD_COLOR)[..., ::-1]
        c_img = cv2.imread(c_f, cv2.IMREAD_COLOR)[..., ::-1]
        vid_img = img_concat([e_img, c_img])
        cv2.putText(vid_img, str(i).zfill(4), (10, c_img.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
        return vid_img

    vid = parallel_map(concat_fn,
                       list(
                           zip(
                               list(range(min(len(e_img_fs), len(c_img_fs)))),
                               e_img_fs, c_img_fs
                           )
                       ), desc="concatenating images", show_pbar=True)
    
    vid = ImageSequenceClip(vid, fps=20)
    vid.write_videofile("/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/lab_c1/trig_stereo.mp4")

if __name__ == "__main__":
    main()