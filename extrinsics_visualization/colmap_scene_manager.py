import glob
import os
import os.path as osp
import cv2
import colmap_find_scale.read_write_model as colmap_utils
import collections
from dataclasses import dataclass
import numpy as np



def proj_3d_pnts(img, intrinsics, extrinsics, pnt_idxs, pnts_3d, dist_coeffs=None):
    """
    Project a list of 3D points onto an image and label them with their indices.
    
    Inputs:
        img: Image array (np.array [h, w, 3])
        intrinsics: Camera intrinsic matrix (np.array [3, 3])
        extrinsics: Camera extrinsic matrix (np.array [4, 4])
        pnt_idxs: Indices of points to be projected (np.array [n])
        pnts_3d: 3D points in world coordinates (np.array [n, 3])
        dist_coeffs: Distortion coefficients (np.array [4, ]) = k1, k2, p1, p2
    Returns:
        proj_pnts: Projected 2D points (np.array [n, 2])
        img_with_pnts: Image with projected points and labels drawn
    """
    # Extract rotation and translation from the extrinsic matrix
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]

    # Filter points based on indices
    selected_pnts_3d = pnts_3d

    # Project points
    proj_pnts_2d, _ = cv2.projectPoints(selected_pnts_3d, R, T, intrinsics, dist_coeffs)

    # Draw points and labels on the image
    img_with_pnts = img.copy()
    for i, p in enumerate(proj_pnts_2d):
        try:
            point = tuple(p[0].astype(int))
            cv2.circle(img_with_pnts, point, 5, (0, 255, 0), -1)
            cv2.putText(img_with_pnts, str(pnt_idxs[i]), (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except Exception as e:
            print("ERROR:", e)
            

    return proj_pnts_2d, img_with_pnts


def draw_2d_pnts(img, pnts_2d, pnt_idxs):
    """
    Draw 2D points and their indices on an image.

    Parameters:
    img (np.array): The image on which to draw the points.
    pnts_2d (np.array): Array of 2D points (n, 2).
    pnt_idxs (np.array): Array of point indices.
    """
    # Create a copy of the image to draw on
    img_with_pnts = img.copy()

    for point, idx in zip(pnts_2d, pnt_idxs):
        # Extract the point coordinates
        x, y = int(point[0]), int(point[1])

        # Draw the point
        cv2.circle(img_with_pnts, (x, y), 5, (0, 255, 0), -1)

        # Draw the index
        cv2.putText(img_with_pnts, str(idx), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img_with_pnts


def concat_imgs(img1, img2):
    """
    Concatenate two images horizontally, padding the shorter image equally on top and bottom to match the height of the taller one.

    Parameters:
    img1 (np.array): First image.
    img2 (np.array): Second image.

    Returns:
    np.array: Concatenated image.
    """
    # Get the heights of the two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Determine the maximum height
    max_height = max(h1, h2)

    # Function to pad an image to the max height
    def pad_image(img, height, width):
        diff = height - img.shape[0]
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        pad_top = np.zeros((pad_top, width, 3), dtype=np.uint8)
        pad_bottom = np.zeros((pad_bottom, width, 3), dtype=np.uint8)
        return np.vstack((pad_top, img, pad_bottom))

    # Pad the images if necessary
    if h1 < max_height:
        img1_padded = pad_image(img1, max_height, w1)
    else:
        img1_padded = img1

    if h2 < max_height:
        img2_padded = pad_image(img2, max_height, w2)
    else:
        img2_padded = img2

    # Concatenate the images horizontally
    concatenated_image = np.hstack((img1_padded, img2_padded))

    return concatenated_image


class Camera:

    def __init__(self, cam_f):
        self.col_cam = colmap_utils.read_cameras_binary(cam_f)[1]
        self.w = self.col_cam.width
        self.h = self.col_cam.height
        
        self.fx, self.fy, self.cx, self.cy = self.col_cam.params[:4]
        self.intrxs = np.array([[self.fx,     0 , self.cx],
                                [0,      self.fy, self.cy],
                                [0,            0,       1]])
        self.k1, self.k2, self.p1, self.p2 = self.col_cam.params[4:]

    def get_dist_coeffs(self):
        return np.array([self.k1, self.k2, self.p1, self.p2])

class ColSceneManager:
    """
    retrieves corresponding 3d points and plot them in 3d
    """
    def __init__(self, colmap_dir, sample_method="reliable"):
        """
        colmap_dir = xxx_recons/
                         /images
                         /sparse
        """
        self.colmap_dir = colmap_dir
        self.img_dir = osp.join(colmap_dir, "images")

        self.img_fs = sorted(glob.glob(osp.join(self.img_dir, "*.png")))
        self.images = colmap_utils.read_images_binary(osp.join(self.colmap_dir, "sparse/0/images.bin"))
        self.pnts_3d = colmap_utils.read_points3D_binary(osp.join(self.colmap_dir,"sparse/0","points3D.bin"))
        self.camera = Camera(osp.join(self.colmap_dir, "sparse/0", "cameras.bin"))
        self.chosen_points = None
        self.pnts_2d=None

        self.sample_pnt_fnc_dic = {"rnd": self.sample_points_rnd,
                                   "reliable": self.sample_pnts_reliable}
        self.sample_method = sample_method
        self.sample_pnt_fnc = self.sample_pnt_fnc_dic[sample_method]

    def set_sample_method(self, method):
        self.sample_method = method
        self.sample_pnt_fnc = self.sample_pnt_fnc_dic[self.sample_method]

    def sample_pnts_reliable(self, img_idx=None, sample_n_points=16):
        # pnt_idxs = self.images[img_idx].point3D_ids
        # val_cond = pnt_idxs != -1
        # val_idxs = self.images[img_idx].point
        # 
        # for idx in val_idxs:
        #     num_imgs.append(len(self.pnts_3d[idx].image_ids))
        # ids = val_idxs_ids[val_cond]

        #####################################################
        ids = []
        num_imgs = []
        for k, pnt in self.pnts_3d.items():
            ids.append(k)
            num_imgs.append(len(pnt.image_ids))
        
        ids = np.array(ids)
        #####################################################

        num_imgs = np.array(num_imgs)
        sorted_idxs = np.argsort(num_imgs)[::-1]
        # return np.random.choice(ids[sorted_idxs[:len(ids)//2]], size=sample_n_points)
        return np.random.choice(ids[sorted_idxs[:len(ids)//4]], size=sample_n_points)



    def sample_points_rnd(self, img_idx, sample_n_points=16):
        pnt_idxs = self.images[img_idx].point3D_ids
        val_cond = pnt_idxs != -1

        val_pnts = self.images[img_idx].xys[val_cond]
        val_idxs = self.images[img_idx].point3D_ids[val_cond]
        chosen_idxs = np.random.choice(len(val_idxs), size=sample_n_points)
        self.chosen_points = {"idxs":val_idxs[chosen_idxs], 
                              "xyzs": self.get_points_xyzs(val_idxs[chosen_idxs])}

        self.pnts_2d = val_pnts
        return self.chosen_points["idxs"]


    def img_to_extrnxs(self, img_idx=None, img_obj=None):
        if img_idx is not None:
            img_obj = self.images[img_idx]
        
        R = img_obj.qvec2rotmat()
        t = img_obj.tvec[..., None]
        mtx = np.concatenate([R,t], axis=-1)
        dummy = np.zeros((1,4))
        dummy[0,-1] = 1
        mtx = np.concatenate([mtx, dummy], axis=0)
        return mtx


    def view_img_points(self, img_idx, rnd=False, sample_n_points = 32, chosen_pnt_idxs = None):
        """
        view points in idx image
        """
        
        if chosen_pnt_idxs is not None:
            chosen_pnt_idxs = chosen_pnt_idxs if type(chosen_pnt_idxs) == np.ndarray else np.array(chosen_pnt_idxs)
        elif self.chosen_points is None:
            chosen_pnt_idxs = self.sample_pnt_fnc(img_idx, sample_n_points=sample_n_points)
        
        self.chosen_points = {"idxs": chosen_pnt_idxs,
                              "xyzs": self.get_points_xyzs(chosen_pnt_idxs)}
        
        img, intrxs, extrxs, pnt_idxs, pnts_3d = cv2.imread(self.img_fs[img_idx]),  \
                                                 self.camera.intrxs, \
                                                 self.img_to_extrnxs(img_idx), \
                                                 self.chosen_points["idxs"], \
                                                 self.chosen_points["xyzs"] 

        img_3d = proj_3d_pnts(np.copy(img), intrxs, extrxs, pnt_idxs, pnts_3d)[1]

        #### debug ####
        # if self.pnts_2d is None:
        #     _, pnt_idxs = self.get_points_xy(img_idx, chosen_pnt_idxs)
        # img_2d = draw_2d_pnts(img, self.pnts_2d, pnt_idxs)

        # comb_img = concat_imgs(img_3d, img_2d)
        #### debug ####
        return img_3d
        

    def get_points_xy(self, img_idx, chosen_idxs):
        pnt_idxs = self.images[img_idx].point3D_ids
        pnt_cond = pnt_idxs == chosen_idxs[0]
        for pnt_idx in chosen_idxs[1:]:
            pnt_cond = pnt_cond | (pnt_idxs == pnt_idx)
        
        assert pnt_cond.sum() > 0

        val_pnts = self.images[img_idx].xys[pnt_cond]

        self.pnts_2d = val_pnts
        return self.pnts_2d, pnt_idxs[pnt_cond]


    def get_points_xyzs(self, pnt_idxs):
        if type(pnt_idxs) == int:
            idxs = np.array([[pnt_idxs]])

        points = np.zeros((len(pnt_idxs), 3))

        for i, idx in enumerate(pnt_idxs):
            points[i] = self.pnts_3d[idx].xyz
        
        return points



if __name__ == "__main__":
    colmap_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/atrium_b2_v1_recons"
    manager = ColSceneManager(colmap_dir)
    manager.view_img_points(3)

