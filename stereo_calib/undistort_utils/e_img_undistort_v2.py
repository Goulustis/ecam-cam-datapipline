import numpy as np
import cv2
from tqdm import tqdm
from typing import Tuple

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

# undistort v1
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


def _compute_residual_and_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
  """Auxiliary function of radial_and_tangential_undistort()."""
  # let r(x, y) = x^2 + y^2;
  #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
  r = x * x + y * y
  d = 1.0 + r * (k1 + r * (k2 + k3 * r))

  # The perfect projection is:
  # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
  # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
  #
  # Let's define
  #
  # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
  # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
  #
  # We are looking for a solution that satisfies
  # fx(x, y) = fy(x, y) = 0;
  fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
  fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

  # Compute derivative of d over [x, y]
  d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
  d_x = 2.0 * x * d_r
  d_y = 2.0 * y * d_r

  # Compute derivative of fx over x and y.
  fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
  fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

  # Compute derivative of fy over x and y.
  fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
  fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

  return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[np.ndarray, np.ndarray]:
  """Computes undistorted (x, y) from (xd, yd)."""
  # Initialize from the distorted point.
  x = xd.copy()
  y = yd.copy()

  for _ in range(max_iterations):
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
        x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    denominator = fy_x * fx_y - fx_x * fy_y
    x_numerator = fx * fy_y - fy * fx_y
    y_numerator = fy * fx_x - fx * fy_x
    step_x = np.where(
        np.abs(denominator) > eps, x_numerator / denominator,
        np.zeros_like(denominator))
    step_y = np.where(
        np.abs(denominator) > eps, y_numerator / denominator,
        np.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

  return x, y

def nerf_undistort(xy, k, distortion, max_iterations = 1):
    k1, k2, p1, p2, k3 = distortion
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]
    x, y = xy.astype(float)
    xd = (x - cx) / fx
    yd = (y - cy) / fy

    new_x, new_y = _radial_and_tangential_undistort(xd, yd, k1, k2, k3, p1, p2, max_iterations=max_iterations)
    new_x = new_x * fx + cx 
    new_y = new_y * fy + cy
    new_x, new_y = new_x - new_x.min(), new_y - new_y.min()
    return np.array((new_x, new_y)).astype(int)
    



def main():
    img_f = "/scratch/matthew/projects/event_calib/stereo_calibration/data/checker_RIGHT/000159.JPG"

    img = cv2.imread(img_f, cv2.IMREAD_GRAYSCALE)
    y, x = img.shape

    x = np.arange(x)
    y = np.arange(y)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.reshape(-1), yy.reshape(-1)
    coords = np.stack([xx, yy])

    # new_xs, new_ys = undistort(coords, intr_mtx, dist)
    new_xs, new_ys = nerf_undistort(coords, intr_mtx, dist)
    
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