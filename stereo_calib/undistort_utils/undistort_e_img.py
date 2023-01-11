import numpy as np
import cv2
from tqdm import tqdm

te_img_f = "/scratch/matthew/projects/event_calib/stereo_calibration/data/checker_RIGHT/000159.JPG"

intr_mtx = np.array([
        [
            1040.2604413841416,
            0.0,
            594.9053486493372
        ],
        [
            0.0,
            1040.5101651348832,
            317.55017718654614
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ])

dist = np.array([
            -0.45493557869168016,
            0.34554615365396835,
            -0.0005402693638344664,
            0.0005747045572511943,
            -0.42255368897693485
        ])

def undistort(xy, k, distortion, iter_num=1):
    k1, k2, p1, p2, k3 = distortion
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]
    x, y = xy.astype(float)
    x = (x - cx) / fx
    x0 = x
    y = (y - cy) / fy
    y0 = y
    for _ in tqdm(range(iter_num)):
        r2 = x ** 2 + y ** 2
        k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
        delta_x = 2 * p1 * x*y + p2 * (r2 + 2 * x**2)
        delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x*y
        x = (x0 - delta_x) * k_inv
        y = (y0 - delta_y) * k_inv
    
    new_x = x * fx + cx 
    new_y = y * fy + cy
    new_x, new_y = new_x - new_x.min(), new_y - new_y.min()

    return np.array((new_x, new_y)).astype(int)
    # return np.array((x * fx + cx, y * fy + cy)).astype(int)

def main():
    img_f = "/scratch/matthew/projects/event_calib/stereo_calibration/data/checker_RIGHT/000159.JPG"

    img = cv2.imread(img_f, cv2.IMREAD_GRAYSCALE)
    y, x = img.shape

    x = np.arange(x)
    y = np.arange(y)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.reshape(-1), yy.reshape(-1)
    coords = np.stack([xx, yy])

    new_xs, new_ys = undistort(coords, intr_mtx, dist)
    
    new_h, new_w = new_ys.max(), new_xs.max()
    new_img = np.zeros((new_h+1, new_w+1))

    for i in tqdm(range(len(new_xs)), desc="mapping"):
        old_x, old_y = xx[i], yy[i]
        new_x, new_y = new_xs[i], new_ys[i]

        new_img[new_y, new_x] = img[old_y, old_x]
    
    new_img = new_img.astype(np.uint8)
    cv2.imwrite("undist.jpg", new_img)
    


if __name__ == "__main__":
    main()