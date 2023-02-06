import sys
sys.path.append(".")

from viz.proj_pts3d_eimgs import read_colmap_cam
import numpy as np
import cv2
import glob
import argparse
from tqdm import tqdm
import json
import os.path as osp
import os

"""
This code is taken from https://github.com/bvnayak/stereo_calibration
and modified
"""

# temporary dir to store intermediate results for sanity check
TMP_DIR = "./tmp"


def save_correspond_img(from_img, to_img, idx):
    save_dir = osp.join(TMP_DIR, "stereo_calib_imgs")
    os.makedirs(save_dir, exist_ok=True)

    expand_dim_fn = lambda img : img if len(img.shape) == 3 else img[...,None]
    from_img, to_img = expand_dim_fn(from_img), expand_dim_fn(to_img)
    save_path = osp.join(save_dir, str(idx).zfill(6) + ".png")
    new_h = max(from_img.shape[0], to_img.shape[0])

    def pad_img(img):
        h, w = img.shape[:2]
        new_img = img
        if (new_h - h) > 0:
            n_add = new_h - h
            n_top = n_add//2
            n_bot = n_add - n_top
            new_img = np.concatenate([np.zeros([n_top] + list(img.shape[1:]), dtype=to_img.dtype), new_img])
            new_img = np.concatenate([new_img, np.zeros([n_bot] + list(img.shape[1:]), dtype=to_img.dtype)])
        
        return new_img
    
    from_img, to_img = pad_img(from_img), pad_img(to_img)
    save_img = np.concatenate([from_img, to_img], axis=1)
    cv2.imwrite(save_path, save_img)


def read_colmap_cam_param(cam_path):
    out = read_colmap_cam(cam_path, scale=1)

    return out["M1"], out["d1"]

def read_prosphesee_ecam_param(cam_path):
    
    with open(cam_path,"r") as f:
        data = json.load(f)
    
    return np.array(data["camera_matrix"]['data']).reshape(3,3), \
           np.array(data['distortion_coefficients']['data'])

class StereoCalibration(object):
    def __init__(self, from_dir, to_dir, grid_size=4.23, n_use=150, st_n=150,
                 colcam_param_path="NULL", ecam_param_path="NULL"):
        """
        grid_size (float): size of grid
        n_use (int): number of frames to use from each camera for calibration
        st_n (int): the frame to start using for calibration (since beginning frames are often bad)
        """
        self.n_use = n_use
        self.st_n = st_n
        self.colcam_param_path = colcam_param_path
        self.ecam_param_path = ecam_param_path
        
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9*6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        self.objp = self.objp*grid_size

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_from = []  # 2d points in image plane.
        self.imgpoints_to = []  # 2d points in image plane.

        # self.cal_path = filepath
        self.from_dir = from_dir
        self.to_dir = to_dir
        self.read_images()

        self.result = None

    def read_images(self):
        sub_sample = lambda x : x[::len(x)//self.n_use]

        images_to = sorted(glob.glob(osp.join(self.to_dir, "*")))[self.st_n:]
        images_from = sorted(glob.glob(osp.join(self.from_dir, "*")))[self.st_n:]

        n_frames = min(len(images_from), len(images_to))
        images_from = images_from[:n_frames]
        images_to = images_to[:n_frames]

        images_to, images_from = sub_sample(images_to), sub_sample(images_from)
        images_from, images_to = images_from[:35], images_to[:35]

        for i in tqdm(range(min(len(images_from), len(images_to))), desc="loading images"):
            img_from = cv2.imread(images_from[i])
            img_to = cv2.imread(images_to[i])

            gray_from = cv2.cvtColor(img_from, cv2.COLOR_BGR2GRAY)
            gray_to = cv2.cvtColor(img_to, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_from, corners_from = cv2.findChessboardCorners(gray_from, (9, 6), None)
            ret_to, corners_to = cv2.findChessboardCorners(gray_to, (9, 6), None)

            # If found, add object points, image points (after refining them)
            if ret_from and ret_to:
                sub_pix_fn = lambda gray_img, corners : cv2.cornerSubPix(gray_img, corners, (11,11),(-1,-1), self.criteria)
                _, _ = sub_pix_fn(gray_from, corners_from), sub_pix_fn(gray_to, corners_to)  # will directly change "corners" memory
                draw_corner = lambda img, corner, ret : cv2.drawChessboardCorners(img, (9,6), corner, ret)
                from_img, to_img  = draw_corner(img_from, corners_from, ret_from), draw_corner(img_to, corners_to, ret_to)
                save_correspond_img(from_img, to_img, i)
                self.objpoints.append(np.array(self.objp))
                self.imgpoints_from.append(corners_from)
                self.imgpoints_to.append(corners_to)

            img_shape = gray_from.shape[::-1]

        print("points available", len(self.objpoints))
        print("calibrating first camera")

        if not osp.exists(self.colcam_param_path):
            rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
                self.objpoints, self.imgpoints_from, gray_from.shape[::-1], None, None)
        else:
            print("reading from colmap intrinsics")
            self.M1, self.d1 = read_colmap_cam_param(self.colcam_param_path)
        
        print("calibrating second camera")
        if not osp.exists(self.ecam_param_path):
            rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
                self.objpoints, self.imgpoints_to, gray_to.shape[::-1], None, None)
        else:
            print("reading from ")
            self.M2, self.d2 = read_prosphesee_ecam_param(self.ecam_param_path)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        print("calibrating stereo camera...")
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_from,
            self.imgpoints_to, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('from_intrinsics', M1)
        print('from_camera_dist', d1)
        print('to_intrinsics', M2)
        print('to_camera_extrinsics', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        print('')

        # camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
        #                     ('dist2', d2), ('rvecs1', self.r1),
        #                     ('rvecs2', self.r2), ('R', R), ('T', T),
        #                     ('E', E), ('F', F)])
        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        return camera_model

if __name__ == '__main__':
    # Description:
    # find R,t such that # t2 = R1@R + t, where R1 is world to cam
    parser = argparse.ArgumentParser(description="find the relative rotation and positions between 2 cameras")
    parser.add_argument("-f", "--from_imgs", help="path to images of camera to find translation from", required=True) # R1
    parser.add_argument("-t", "--to", help="path to images of camera loc to map to", required=True) # R2
    parser.add_argument("-s", "--size", type=float, help="size of the checkerboard square", default=4.23)
    parser.add_argument("-o", "--out", help="location to save the relative camera output", default=None)
    parser.add_argument("-cp", "--colcam_param", help="path to colmap param binary file", default="None")
    parser.add_argument("-ep", "--ecam_param", help="path to prosphesee event camera param path", default="None")
    # parser.add_argument("-f", "--from_imgs", help="path to images of camera to find translation from", default="data/rgb_checker/rgb_checker_recon/images") # R1
    # parser.add_argument("-t", "--to", help="path to images of camera loc to map to", default="data/rgb_checker/events_imgs") # R2
    # parser.add_argument("-s", "--size", type=float, help="size of the checkerboard square", default=4.28)
    # parser.add_argument("-o", "--out", help="location to save the relative camera output", default=None)
    # parser.add_argument("-cp", "--colcam_param", help="path to colmap param binary file", default="data/rgb_checker/rgb_checker_recon/sparse/0/cameras.bin")
    # parser.add_argument("-ep", "--ecam_param", help="path to prosphesee event camera param path", default="data/rgb_checker/intrinsics.json")
    args = parser.parse_args()

    if args.out is None:
        out_path = osp.join(osp.dirname(args.to), "rel_cam.json")
    else:
        out_path = args.out
    # cal_data = StereoCalibration(args.from_imgs, args.to, args.size, colcam_param_path=args.colcam_param, ecam_param_path=args.ecam_param)
    cal_data = StereoCalibration(args.from_imgs, args.to, args.size, 
                                 colcam_param_path="data/rgb_checker/rgb_checker_recon/sparse/0/cameras.bin", 
                                 ecam_param_path="data/rgb_checker/intrinsics.json")
    cam_model = cal_data.camera_model

    to_save = {}
    for k, v in cam_model.items():
        if type(v) == np.ndarray:
            to_save[k] = v.tolist()

    with open(out_path, "w") as f:
        json.dump(to_save, f, indent=4)