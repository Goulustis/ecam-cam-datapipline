import os.path as osp
import glob
import cv2
from concurrent import futures
import numpy as np
from tqdm import tqdm 
import json
import shutil
import os
import argparse


def parallel_map(f, iterable, max_threads=None, show_pbar=False, desc="", **kwargs):
  """Parallel version of map()."""
  with futures.ThreadPoolExecutor(max_threads) as executor:
    if show_pbar:
      results = tqdm(
          executor.map(f, iterable, **kwargs), total=len(iterable), desc=desc)
    else:
      results = executor.map(f, iterable, **kwargs)
    return list(results)


def save_imgs_cams(imgs, cam_fs, dst_dir):
    os.makedirs(osp.join(dst_dir, "camera"), exist_ok=True)
    os.makedirs(osp.join(dst_dir, "rgb/1x"), exist_ok=True)
    
    for i, (img, cam_f) in tqdm(enumerate(zip(imgs, cam_fs)), total=len(imgs), desc="saving imgs, cams"):
        dst_img_f = osp.join(dst_dir, "rgb/1x",f"{str(i).zfill(4)}.png")
        dst_cam_f = osp.join(dst_dir, "camera" ,f"{str(i).zfill(4)}.json")
        save_flag = cv2.imwrite(dst_img_f, np.clip(0,255,img).astype(np.uint8))
        assert save_flag, "save failed! go debug"
        shutil.copy(cam_f, dst_cam_f)        


def modify_save_metadata(src_dir, dst_dir, cam_fs):
    meta_f = osp.join(src_dir, "metadata.json")
    with open(meta_f, "r") as f:
        data = json.load(f)
    
    want_keys = [osp.basename(cam_f).split(".")[0] for cam_f in cam_fs]
    new_meta = {}
    for k in want_keys:
        new_meta[k] = data[k]

    with open(osp.join(dst_dir, "metadata.json"), "w") as f:
        json.dump(new_meta, f, indent=2)

def modify_save_dataset(src_dir, dst_dir):
    total_frames = len(glob.glob(osp.join(dst_dir, "1x/*.json")))
    ids = [osp.basename(e).split(".")[0] for e in sorted(glob.glob(osp.join(dst_dir, "camera/*.json")))]
    dataset_json = {"counts":total_frames,
                    "num_exemplars": total_frames,
                    "ids":ids,
                    "train_ids":ids
                    }
    
    with open(osp.join(dst_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)


def main(src_dir, num_blur=8):
    dst_dir = osp.join(osp.dirname(src_dir), "blur_gamma_colcam_set")
    print("saving to:", dst_dir)

    cam_fs = sorted(glob.glob(osp.join(src_dir, "camera/*.json")))
    img_fs = sorted(glob.glob(osp.join(src_dir,"rgb/1x", "*.png")))[:len(cam_fs)]

    
    imgs = np.stack(parallel_map(lambda x : cv2.imread(x), img_fs, show_pbar=True, desc="loading imgs"))
    
    blur_imgs = np.zeros((int(len(cam_fs)//num_blur), *imgs[0].shape), dtype=np.float32)
    mid_idxs = []
    for i in tqdm(range(int(len(cam_fs)/num_blur)), desc="blurring..."):
        blur_img = (imgs[i*num_blur:(i+1)*num_blur].astype(np.float32)).mean(axis=0)
        blur_imgs[i] = blur_img
        mid_idxs.append((i*num_blur+(i+1)*num_blur)/2)
    
    cam_fs = cam_fs[int(num_blur//2)::num_blur]

    save_imgs_cams(blur_imgs, cam_fs, dst_dir)
    modify_save_metadata(src_dir, dst_dir, cam_fs)
    modify_save_dataset(src_dir, dst_dir)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create blur colcamset from clear colcam_set")
    parser.add_argument("--src_dir", help="path to xxx/colcam_set", default="/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/book_sofa/colcam_set")
    parser.add_argument("--num_blur", help="number of frames to average", type=int, default=8)

    args = parser.parse_args()
    main(args.src_dir, args.num_blur)