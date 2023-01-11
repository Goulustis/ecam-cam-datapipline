import os
import os.path as osp
import matplotlib.pyplot as plt

from PIL import Image
import glob
from tqdm import tqdm

## turn pmg files to JPG; since JPG is used by the algorithm


def process_img_files(img_fs, save_dir):
    """
    img_fs (list [string]): list of image files, sort them and save them as jpg
    """
    os.makedirs(save_dir, exist_ok=True)

    img_fs = sorted(img_fs)


    for i, f in tqdm(enumerate(img_fs), total=len(img_fs)):
        # f_name = f.split("/")[-1].split(".")[0]
        f_name = str(i).zfill(6) + ".JPG"
        target_path = osp.join(save_dir, f_name)

        img = Image.open(f)
        img.save(target_path)


        
def main():
    e_path = "checker_event"
    c_path = "checker"

    e_fs = glob.glob(osp.join(e_path, "*")) # event image file paths
    c_fs = glob.glob(osp.join(c_path, "*")) # camera image file paths

    process_img_files(c_fs, c_path + "_" + "LEFT")
    process_img_files(e_fs, c_path + "_" + "RIGHT")


class Viz:
    def __init__(self, left_dir, right_dir):
        self.l_dir = left_dir
        self.r_dir = right_dir

        self.l_fs = sorted(glob.glob(osp.join(self.l_dir, "*")))
        self.r_fs = sorted(glob.glob(osp.join(self.r_dir, "*")))

    def __len__(self):
        return min(len(self.l_fs), len(self.r_fs))

    def __getitem__(self, idx):
        l_f = self.l_fs[idx]
        r_f = self.r_fs[idx]

        fig, axes = plt.subplots(1,2)

        imgl, imgr = plt.imread(l_f), plt.imread(r_f)
        axes[0].imshow(imgl)
        axes[1].imshow(imgr)

        axes[0].set_title("camera img")
        axes[1].set_title("event img")

        plt.show()


def get_viz():
    return Viz("./LEFT", "./RIGHT")


if __name__ == "__main__":
    main()

