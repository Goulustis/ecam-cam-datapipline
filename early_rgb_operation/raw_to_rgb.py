from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import os
import os.path as osp
import glob
import cv2
from concurrent import futures
from functools import partial

H, W = 1080, 1440

def parallel_map(f, iterable, max_threads=None, show_pbar=False, desc="", **kwargs):
  """Parallel version of map()."""
  with futures.ThreadPoolExecutor(max_threads) as executor:
    if show_pbar:
      results = tqdm(
          executor.map(f, iterable, **kwargs), total=len(iterable), desc=desc)
    else:
      results = executor.map(f, iterable, **kwargs)
    return list(results)

def raw_to_rgb(raw_f):
    with open(raw_f, "rb") as f:
       img_data = np.frombuffer(f.read(), np.uint16, H*W).reshape(H, W)
       img = cv2.cvtColor(img_data, cv2.COLOR_BAYER_BG2BGR)
       img = cv2.convertScaleAbs(img, alpha=(255/65535))
    
    return img

def transform_img_and_save(inp, dst_dir):
    """
    inp = (raw_f:str, img_idx:int)
    """

    raw_f, img_idx = inp

    dst_f = osp.join(dst_dir, f"{str(img_idx).zfill(5)}.png")
    img = raw_to_rgb(raw_f)
    save_stat = cv2.imwrite(dst_f, img)
    assert save_stat, "save failed !"


def main(input_dir):
    raw_fs = sorted(glob.glob(osp.join(input_dir, "*.raw")))

    dst_dir = osp.join(input_dir + "_recons", "images")
    os.makedirs(dst_dir, exist_ok=True)
    map_fn = partial(transform_img_and_save, dst_dir=dst_dir)
    parallel_map(map_fn, list(zip(raw_fs, range(len(raw_fs)))), show_pbar=True, desc="processing raw")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="scratch")
    args = parser.parse_args()
    main(args.input_dir)