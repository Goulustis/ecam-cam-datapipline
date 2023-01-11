import numpy as np
import cv2
import glob
import argparse
from tqdm import tqdm
import json
import os.path as osp

"""
This code is taken from https://github.com/bvnayak/stereo_calibration
and modified
"""

class StereoCalibration(object):
    def __init__(self, from_dir, to_dir, grid_size=4.23, n_use=150, st_n=150):
        """
        grid_size (float): size of grid
        n_use (int): number of frames to use from each camera for calibration
        st_n (int): the frame to start using for calibration (since beginning frames are often bad)
        """
        self.n_use = n_use
        self.st_n = st_n
        
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

        # for i, fname in tqdm(enumerate(images_right), total=len(images_right)):
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
                self.objpoints.append(np.array(self.objp))
                self.imgpoints_from.append(corners_from)
                self.imgpoints_to.append(corners_to)

            img_shape = gray_from.shape[::-1]

        print("points available", len(self.objpoints))
        print("calibrating first camera")
        # rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
        #     self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_from, gray_from.shape[::-1], None, None)
        
        print("calibrating second camera")
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_to, gray_to.shape[::-1], None, None)

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

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        return camera_model

if __name__ == '__main__':
    # Description:
    # find R,t such that # R2 = R1@R + t, where R1 is world to cam
    parser = argparse.ArgumentParser(description="find the relative rotation and positions between 2 cameras")
    parser.add_argument("-f", "--from", help="path to images of camera to find translation from", required=True) # R1
    parser.add_argument("-t", "--to", help="path to images of camera loc to map to", required=True) # R2
    parser.add_argument("-s", "--size", help="size of the checkerboard square", default=4.23)
    parser.add_argument("-o", "--out", help="location to save the relative camera output", default=None)
    args = parser.parse_args()

    if args.o is None:
        out_path = osp.join(osp.dirname(args.t), "rel_cam.json")
    else:
        out_path = args.o
    cal_data = StereoCalibration(args.f, args.to, args.size)
    cam_model = cal_data.camera_model

    to_save = {}
    for k, v in cam_model.items():
        if type(v) == np.ndarray:
            to_save[k] = v.tolist()

    with open(out_path, "w") as f:
        json.dump(to_save, f, indent=4)