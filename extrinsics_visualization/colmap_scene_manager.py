import glob
from utils.misc import parallel_map
import os
import os.path as osp
import cv2
import colmap_find_scale.read_write_model as colmap_utils
import collections
from dataclasses import dataclass
import numpy as np
from nerfies.camera import Camera



def proj_3d_pnts(img, intrinsics, extrinsics, pnts_3d, pnt_idxs=None, dist_coeffs=None):
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
    if img is not None:
        try:
            img_with_pnts = draw_2d_pnts(img, proj_pnts_2d)
        except:
            print("projection error, but ignoring")
            img_with_pnts = img
    else:
        img_with_pnts = img

    return proj_pnts_2d, img_with_pnts


def draw_2d_pnts(img, pnts_2d, pnt_idxs=None):
    """
    Draw 2D points and their indices on an image.

    Parameters:
    img (np.array): The image on which to draw the points.
    pnts_2d (np.array): Array of 2D points (n, 2).
    pnt_idxs (np.array): Array of point indices.
    """
    pnts_2d = pnts_2d.squeeze()
    pnts_2d = pnts_2d[None] if len(pnts_2d.shape) == 1 else pnts_2d
    pnt_idxs = list(range(len(pnts_2d))) if pnt_idxs is None else pnt_idxs
    # Create a copy of the image to draw on
    img_with_pnts = img.copy()

    for point, idx in zip(pnts_2d, pnt_idxs):
        # Extract the point coordinates
        x, y = int(point[0]), int(point[1])

        # Draw the point
        cv2.circle(img_with_pnts, (x, y), 3, (0, 255, 0), -1)

        # Draw the index
        cv2.putText(img_with_pnts, str(idx), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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


class ColmapCamera:

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

class ColmapSceneManager:
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
        self.camera = ColmapCamera(osp.join(self.colmap_dir, "sparse/0", "cameras.bin"))
        self.chosen_points = None
        self.pnts_2d=None

        self.max_images_id = max(list(self.images.keys()))

    def set_sample_method(self, method):
        self.sample_method = method
        self.sample_pnt_fnc = self.sample_pnt_fnc_dic[self.sample_method]
    
    @property
    def image_ids(self):
        return sorted(self.images.keys())
    
    def load_image(self, img_idx):
        img_cam = self.images[img_idx]
        img_name = img_cam.name # get img_name from img_cam
        img_f = osp.join(self.img_dir, img_name)
        return cv2.imread(img_f)



    def __len__(self):
        return len(self.images)


    def get_extrnxs(self, img_idx=None, img_obj=None):
        """
        img_idx: colmap index
        """
        if img_idx < 0:
            img_idx = self.max_images_id + img_idx
            
        if img_idx is not None:
            img_obj = self.images[img_idx]
        
        R = img_obj.qvec2rotmat()
        t = img_obj.tvec[..., None]
        mtx = np.concatenate([R,t], axis=-1)
        dummy = np.zeros((1,4))
        dummy[0,-1] = 1
        mtx = np.concatenate([mtx, dummy], axis=0)
        return mtx

    def get_all_extrnxs(self):
        keys = sorted(list(self.images.keys()))
        return [self.get_extrnxs(e) for e in keys]

    def get_intrnxs(self):
        return self.camera.intrxs, self.camera.get_dist_coeffs()

    def get_img(self, img_idx):
        """
        img_idx = colmap_img_idx
        """
        if img_idx > 0: 
            print("colmap img is 1 indexed! Adding 1 to idx")
            img_idx += 1
        return cv2.imread(self.get_img_f(img_idx))

    def get_all_imgs(self, read_img_fn=cv2.imread):
        return parallel_map(lambda f : read_img_fn(f), self.img_fs)

    def get_img_f(self, img_idx):
        if img_idx == 0:
            print("image_idx needs to be >=1, adding 1 as fix")
            img_idx += 1
        if img_idx < 0:
            img_idx = self.max_images_id + img_idx
        return osp.join(self.img_dir, self.images[img_idx].name)

        

    def get_points_xy(self, img_idx, chosen_idxs):
        pnt_idxs = self.images[img_idx].point3D_ids
        pnt_cond = pnt_idxs == chosen_idxs[0]
        for pnt_idx in chosen_idxs[1:]:
            pnt_cond = pnt_cond | (pnt_idxs == pnt_idx)
        
        assert pnt_cond.sum() > 0

        val_pnts = self.images[img_idx].xys[pnt_cond]

        self.pnts_2d = val_pnts
        return self.pnts_2d, pnt_idxs[pnt_cond]


    def get_img_id(self, img_idx):
        img_f = self.get_img_f(img_idx)
        return osp.basename(img_f).split(".")[0]
    

    def get_found_cond(self, n_size):
        """
        return condition showing which image was registered
        """
        keys = np.array(sorted(list(int(k) for k in self.images.keys()))) - 1 # subtract 1 since colmap idx starts at 1
        keys = keys[keys < n_size].astype(np.int32)
        cond = np.zeros(n_size, dtype=bool)
        cond[keys] = True
        return cond

class ColcamSetManager:

    def __init__(self, colcam_set_dir):
        self.colcam_set_dir = colcam_set_dir
        self.img_fs = sorted(glob.glob(osp.join(colcam_set_dir, "rgb", "1x", "*.png")))
        self.cam_fs = sorted(glob.glob(osp.join(colcam_set_dir, "camera", "*.json")))

        self.ref_cam = Camera.from_json(self.cam_fs[0])

    def __len__(self):
        return len(self.img_fs)

    def get_img_f(self,idx):
        return self.img_fs[idx]

    def get_extrnxs(self, idx):
        extrxs_f = self.cam_fs[idx]
        cam = Camera.from_json(extrxs_f)
        R = cam.orientation
        T = -R@cam.position
        T = T.reshape(3,1)

        return np.concatenate([R, T], axis=1)

    def get_intrnxs(self):
        """
        return K, distortions
        """
        fx = fy = self.ref_cam.focal_length
        cx, cy = self.ref_cam.principal_point_x, self.ref_cam.principal_point_y
        k1, k2, k3 = self.ref_cam.radial_distortion
        p1, p2 = self.ref_cam.tangential_distortion

        intrx_mtx = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
        dist = np.array((k1, k2, p1, p2))

        return intrx_mtx, dist

if __name__ == "__main__":
    colmap_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/atrium_b2_v1_recons"
    manager = ColmapSceneManager(colmap_dir)
    manager.view_img_points(3)

