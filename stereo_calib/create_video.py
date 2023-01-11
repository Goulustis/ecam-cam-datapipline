from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np
import cv2
import glob
import os.path as osp
from tqdm import tqdm

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
    e_img_fs = sorted(glob.glob(osp.join("data/checker_RIGHT", "*")))
    c_img_fs = sorted(glob.glob(osp.join("data/checker_LEFT", "*")))
    n_frames = min(len(e_img_fs), len(c_img_fs))

    e_img_fs = e_img_fs[:n_frames]
    c_img_fs = c_img_fs[:n_frames]


    vid = []
    for i, (e_f, c_f) in tqdm(enumerate(zip(e_img_fs, c_img_fs)), desc="making vid", total=n_frames):
        e_img = cv2.imread(e_f, cv2.IMREAD_COLOR)#[..., None]
        c_img = cv2.imread(c_f, cv2.IMREAD_COLOR)#[..., None]
        vid_img = img_concat([e_img, c_img])
        cv2.putText(vid_img, str(i).zfill(4), (10, c_img.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
        vid.append(vid_img)
    
    vid = ImageSequenceClip(vid, fps=55)
    vid.write_videofile("stereo.mp4")

if __name__ == "__main__":
    main()