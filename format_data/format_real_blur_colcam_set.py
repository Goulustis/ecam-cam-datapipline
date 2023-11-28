import os.path as osp
import os

import glob
import numpy as np
import shutil
from tqdm import tqdm
import argparse
import imageio
from concurrent import futures
from tqdm import tqdm
import scipy.ndimage as ndimage
import scipy.signal as signal

from format_data.format_avg_blur_colcam_set import modify_save_metadata, modify_save_dataset


def parallel_map(f, iterable, max_threads=None, show_pbar=False, desc="", **kwargs):
  """Parallel version of map()."""
  with futures.ThreadPoolExecutor(max_threads) as executor:
    if show_pbar:
      results = tqdm(
          executor.map(f, iterable, **kwargs), total=len(iterable), desc=desc)
    else:
      results = executor.map(f, iterable, **kwargs)
    return list(results)
  
def calc_clearness_score(img_list, ignore_first = 0):
    # Get list of images in folder
    img_list = img_list[ignore_first:]

    # Load images
    images = parallel_map(imageio.imread, img_list, show_pbar=True, desc="loading imgs")

    blur_scores = []
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    blur_kernels = np.array([[
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ], [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], [
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ]], dtype=np.float32) / 5.0
    for image in tqdm(images, desc="caculating blur score"):
        gray_im = np.mean(image, axis=2)[::4, ::4]

        directional_blur_scores = []
        for i in range(4):
            blurred = ndimage.convolve(gray_im, blur_kernels[i])

            laplacian = signal.convolve2d(blurred, laplacian_kernel, mode="valid")
            var = laplacian**2
            var = np.clip(var, 0, 1000.0)

            directional_blur_scores.append(np.mean(var))

        antiblur_index = (np.argmax(directional_blur_scores) + 2) % 4

        blur_score = directional_blur_scores[antiblur_index]
        blur_scores.append(blur_score)
    
    ids = np.argsort(blur_scores) + ignore_first
    best = ids[::-1]
 
    clear_image_fs = [img_list[e] for e in best]
    return clear_image_fs, best, np.array(blur_scores)


def save_imgs_cams(img_fs, cam_fs, dst_dir):
    os.makedirs(osp.join(dst_dir, "camera"), exist_ok=True)
    os.makedirs(osp.join(dst_dir, "rgb/1x"), exist_ok=True)
    
    for i, (img_f, cam_f) in tqdm(enumerate(zip(img_fs, cam_fs)), total=len(img_fs), desc="saving imgs, cams"):
        dst_img_f = osp.join(dst_dir, "rgb/1x",f"{str(i).zfill(4)}.png")
        dst_cam_f = osp.join(dst_dir, "camera" ,f"{str(i).zfill(4)}.json")
        shutil.copy(img_f, dst_img_f)
        shutil.copy(cam_f, dst_cam_f)     


def main(src_dir, thresh=56):
    # src_dir = ""

    dst_dir = osp.join(osp.dirname(src_dir), "real_blur_gamma")

    os.makedirs(dst_dir, exist_ok=True)

    cam_fs = sorted(glob.glob(osp.join(src_dir, "camera/*.json")))
    img_fs = sorted(glob.glob(osp.join(src_dir, "rgb", "1x", "*.png")))[:len(cam_fs)]

    scores = calc_clearness_score(img_fs)[-1]

    idxs = np.arange(len(scores))[scores <= thresh]
    img_fs = [img_fs[e] for e in idxs]
    cam_fs = [cam_fs[e] for e in idxs]

    save_imgs_cams(img_fs, cam_fs, dst_dir)
    modify_save_metadata(src_dir, dst_dir, cam_fs)
    modify_save_dataset(src_dir, dst_dir)

    src_cam_transform_f = osp.join(src_dir, "camera_transform.json")
    if osp.exists(src_cam_transform_f):
        shutil.copy(src_cam_transform_f, dst_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create blur colcamset from clear colcam_set")
    parser.add_argument("--src_dir", help="path to xxx/colcam_set", default="/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/halloween_b1_v1/colcam_set")
    parser.add_argument("--thresh", help="thresh of blur score, lower the blurrier", type=float, default=56)

    args = parser.parse_args()
    main(args.src_dir, args.thresh)