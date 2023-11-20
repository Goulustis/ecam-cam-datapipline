import argparse
import glob
import os.path as osp

def create_del_img_txt(img_dir, n_keep=200):
    print("creating img list to del")
    fs = sorted(glob.glob(osp.join(img_dir, "*.png")))
    delta = len(fs)//n_keep

    to_keeps = [fs[idx] for idx in range(0,len(fs), delta)]
    to_dels = sorted(list(set(fs) - set(to_keeps)))

    print(len(set(fs) - set(to_dels)))
    save_txt = osp.join(osp.dirname(img_dir), "del_paths.txt")
    with open(save_txt, "w") as f:
        for e in to_dels:
            f.write(osp.basename(e) + "\n")
    
    print(save_txt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/atrium_b2_v1_recons/images")

    args = parser.parse_args()
    create_del_img_txt(args.img_dir)