
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os.path as osp
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
import concurrent

import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument("--eimg_f")
# parser.add_argument("--sparse_txt_dir")
# parser.add_argument("--dense_dir")
# parser.add_argument("--save_dir")

# args = parser.parse_args()

# ev_img_f = args.ev_img_f
# sparse_txt_dir = args.sparse_txt_dir
# dense_dir = args.dense_dir
# save_dir = args.save_dir

#ev_img_f = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/halloween_b2_v1/ecam_set/eimgs/eimgs_1x.npy"
#sparse_txt_dir = "/scratch-ssd/dense_workdir/halloween_b2_v1_recons/dense/txt_sparse"
#dense_dir = "/scratch-ssd/dense_workdir/halloween_b2_v1_recons/dense"
#save_dir = "vid_frames"

ev_img_f = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/calib_checker/ecam_set/eimgs/eimgs_1x.npy"
sparse_txt_dir = "/scratch-ssd/dense_workdir/calib_checker_recons/dense/txt_sparse"
dense_dir = "/scratch-ssd/dense_workdir/calib_checker/dense"
save_dir = "calib_checker_vid_frames"


eimgs = np.load(ev_img_f, "r")


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()



plt.rc("figure", figsize=(15, 15))



def quaternion_to_rotation_matrix(q):
    q = np.array(q)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)

    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    matrix = np.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                       txy + twz, 1.0 - (txx + tzz), tyz - twx,
                       txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                      axis=-1)  # pyformat: disable
    output_shape = q.shape[:-1] + (3, 3)
    return np.reshape(matrix, output_shape)



with open(osp.join(sparse_txt_dir, "cameras.txt")) as fd:
    cam_txt = fd.read()
    im_w, im_h = [int(s) for s in cam_txt.split("\n")[3].split()[2:4]]
    fx, fy, ppx, ppy = [float(s) for s in cam_txt.split("\n")[3].split()[4:8]]
    focal = np.array([[fx, fy]])
    principal = np.array([[ppx, ppy]])

id_to_index = {}
rotations = []
translations = []
images = []
depth_images = []
depth_inds = []
depth_vals = []
    

with open(osp.join(sparse_txt_dir, "images.txt")) as fd:
    lines = fd.read().split("\n")
    for i in tqdm(range((len(lines) - 4) // 2), desc="loading colmap imgs"):
        line = lines[4 + 2 * i].split()
        

        if len(line) < 10 or not osp.exists(osp.join(dense_dir, "stereo/depth_maps/{}.geometric.bin").format(line[9])):
            continue
        
        id_to_index[int(line[0])] = i
        
        rotation = quaternion_to_rotation_matrix(list(float(line[j]) for j in [2, 3, 4, 1]))
        rotations.append(rotation)
        
        translation = np.transpose(rotation, (1, 0)) @ np.reshape(list(float(s) for s in line[5:8]), (3, 1))
        translations.append(translation[:, 0])
        

        im = plt.imread(osp.join(dense_dir, "images", line[9]))
        images.append(im)
        
        depth_im = read_array(osp.join(dense_dir, "stereo/depth_maps/{}.geometric.bin").format(line[9]))
        depth_images.append(depth_im)
        
        inds = np.argwhere(np.logical_and(depth_im > 1.0, depth_im < 8.0))
        depth_inds.append(inds)
        depth_vals.append(depth_im[inds[..., 0], inds[..., 1]])

translations = np.stack(translations, axis=0)
rotations = np.stack(rotations, axis=0)

points = []
point_rgb = []


with open(osp.join(sparse_txt_dir, "points3D.txt")) as fd:
    lines = fd.read().split("\n")
    for l in tqdm(lines[4:-1], desc="loading points3d"):
        line = l.split()
        points.append(list(float(s) for s in line[1:4]))
        point_rgb.append(list(float(s) / 255.0 for s in line[4:7]))
        
points = np.array(points)
point_rgb = np.array(point_rgb)


q = 0
n = 3

rgb_q = images[q]
rgb_n = images[n]

depth_q = depth_images[q]
depth_n = depth_images[n]

rotation_q = rotations[q]
rotation_n = rotations[n]

translation_q = translations[q]
translation_n = translations[n]


def unproject_points(screen_space_points, depths, focal, principal):
    points = np.copy(screen_space_points)
    points[..., 0] = (points[..., 0] - principal[0, 0]) / focal[0, 0]
    points[..., 1] = (points[..., 1] - principal[0, 1]) / focal[0, 1]
    points = points * depths[..., None]
    return np.concatenate([points, depths[..., None]], axis=-1)

def project_points(points, focal, principal):
    screen_space_points = np.copy(points)
    screen_space_points[..., 0] = screen_space_points[..., 0] / screen_space_points[..., 2] * focal[0, 0] + principal[0, 0]
    screen_space_points[..., 1] = screen_space_points[..., 1] / screen_space_points[..., 2] * focal[0, 1] + principal[0, 1]
    return screen_space_points[..., :2]



point_inds_q = np.argwhere(depth_q >= 1e-1)
point_inds_n = np.argwhere(depth_n >= 1e-1)

np.random.shuffle(point_inds_q)
np.random.shuffle(point_inds_n)

focal_q = np.tile(focal, (point_inds_q.shape[0], 1))
focal_n = np.tile(focal, (point_inds_n.shape[0], 1))

principal_q = np.tile(principal, (point_inds_q.shape[0], 1))
principal_n = np.tile(principal, (point_inds_n.shape[0], 1))

point_depth_n = depth_n[point_inds_n[..., 0], point_inds_n[..., 1]]

point_rgb_n = rgb_n[point_inds_n[..., 0], point_inds_n[..., 1]]

point_ss_n = np.asarray(point_inds_n, dtype=np.float32)[..., ::-1]

point_vs_n = unproject_points(point_ss_n, point_depth_n, focal_n, principal_n)


point_ws_n = (point_vs_n @ rotation_n) - translation_n



rotation_gt = rotation_n @ np.transpose(rotation_q, (1, 0))
translation_gt = (translation_n - translation_q)[None] @ np.transpose(rotation_q, (1, 0))

subsample = 1

transformed_points = point_ws_n @ np.transpose(rotation_q, (1, 0)) + (translation_q @ np.transpose(rotation_q, (1, 0)))
points_rgb = point_rgb_n

ss_reproj = project_points(transformed_points, focal, principal)[::subsample]
mask = (ss_reproj[..., 0] >= 0) & (ss_reproj[..., 0] < im_w) & (ss_reproj[..., 1] >= 0) & (ss_reproj[..., 1] < im_h)
ss_reproj = ss_reproj[mask]
inds = np.asarray(np.floor(ss_reproj), dtype=np.int32)

scatter_im = np.zeros((im_h, im_w, 3))
scatter_im[inds[..., 1], inds[..., 0]] = points_rgb[::subsample][mask]

plt.figure(figsize=(15,15))
plt.imshow(scatter_im * 0.5 + rgb_q * 0.5)

plt.figure(figsize=(15,15))
plt.imshow(scatter_im)

# eimg_inds = np.arange(0, 960)
eimg_inds = np.arange(0, len(eimgs))

##################################################
# eimg_rotations = []
# eimg_translations = []
# eimg_focals = []
# eimg_principals = []
# eimg_radial_distortions = []
# eimg_tangential_distortions = []

# for i in tqdm(eimg_inds, desc="loading eimg cams"):
#     with open(osp.join(osp.dirname(osp.dirname(ev_img_f)), f"camera/{i:06d}.json")) as fd:
#         camera = json.load(fd)
    
#     eimg_rotations.append(np.array(camera["orientation"]))
#     eimg_translations.append(np.array(camera["position"]))
#     eimg_focals.append(np.array(camera["focal_length"]))
#     eimg_principals.append(np.array(camera["principal_point"]))
#     eimg_radial_distortions.append(np.array(camera["radial_distortion"]))
#     eimg_tangential_distortions.append(np.array(camera["tangential_distortion"]))

# eimg_rotations = np.stack(eimg_rotations, axis=0)
# eimg_translations = np.stack(eimg_translations, axis=0)
# eimg_focals = np.stack(eimg_focals, axis=0)
# eimg_principals = np.stack(eimg_principals, axis=0)
# eimg_radial_distortions = np.stack(eimg_radial_distortions, axis=0)
# eimg_tangential_distortions = np.stack(eimg_tangential_distortions, axis=0)
##################################################
def load_camera_data(index):
    with open(osp.join(osp.dirname(osp.dirname(ev_img_f)), f"camera/{index:06d}.json")) as fd:
        camera = json.load(fd)

    return {
        "rotation": np.array(camera["orientation"]),
        "translation": np.array(camera["position"]),
        "focal_length": np.array(camera["focal_length"]),
        "principal_point": np.array(camera["principal_point"]),
        "radial_distortion": np.array(camera["radial_distortion"]),
        "tangential_distortion": np.array(camera["tangential_distortion"])
    }

# Initialize lists to hold the data
eimg_rotations, eimg_translations, eimg_focals, eimg_principals, eimg_radial_distortions, eimg_tangential_distortions = ([] for _ in range(6))

# Use ThreadPoolExecutor to parallelize the file reading
with ThreadPoolExecutor() as executor:
    # Prepare a list of futures
    tasks = [executor.submit(load_camera_data, i) for i in eimg_inds]

    # Iterate over the futures as they complete
    for future in tqdm(concurrent.futures.as_completed(tasks), desc="Processing", total=len(tasks)):
        data = future.result()
        eimg_rotations.append(data["rotation"])
        eimg_translations.append(data["translation"])
        eimg_focals.append(data["focal_length"])
        eimg_principals.append(data["principal_point"])
        eimg_radial_distortions.append(data["radial_distortion"])
        eimg_tangential_distortions.append(data["tangential_distortion"])

# Stack the data
eimg_rotations = np.stack(eimg_rotations, axis=0)
eimg_translations = np.stack(eimg_translations, axis=0)
eimg_focals = np.stack(eimg_focals, axis=0)
eimg_principals = np.stack(eimg_principals, axis=0)
eimg_radial_distortions = np.stack(eimg_radial_distortions, axis=0)
eimg_tangential_distortions = np.stack(eimg_tangential_distortions, axis=0)


def eimg_to_rgb(eimg):
    eimg = np.asarray(eimg, dtype=np.float32)
    eimg = np.tile(5 * np.abs(eimg)[..., None], (1, 1, 3))
    eimg = np.clip(eimg, 0, 1)
    return eimg

os.makedirs(save_dir, exist_ok=True)



def parallel_map(f, iterable, max_threads=None, show_pbar=False, desc="", **kwargs):
  """Parallel version of map()."""
  with futures.ThreadPoolExecutor(max_threads) as executor:
    if show_pbar:
      results = tqdm(
          executor.map(f, iterable, **kwargs), total=len(iterable), desc=desc)
    else:
      results = executor.map(f, iterable, **kwargs)
    return list(results)


def projection_image(inp):
    i, index = inp
    rotation_q = eimg_rotations[index]
    translation_q = eimg_translations[index]
    focal_q = np.tile(eimg_focals[index][None, None], (1, 2))
    principal_q = eimg_principals[index][None]
    radial_distortion_q = eimg_radial_distortions[index]
    tangential_distortion_q = eimg_tangential_distortions[index]
    rgb_q = eimg_to_rgb(eimgs[eimg_inds[index]])

    im_h, im_w = rgb_q.shape[:2]

    translation_q = -translation_q

    transformed_points = point_ws_n @ np.transpose(rotation_q, (1, 0)) + (translation_q @ np.transpose(rotation_q, (1, 0)))
    points_rgb = point_rgb_n


    rvec = cv2.Rodrigues(np.eye(3))[0]
    tvec = np.zeros((3, ))
    cam_matrix = np.array([[focal_q[0, 0], 0, principal_q[0, 0]], [0, focal_q[0, 1], principal_q[0, 1]], [0, 0, 1]])
    dist_coeffs = np.concatenate([radial_distortion_q[:2], tangential_distortion_q], axis=-1)
    ss_reproj = cv2.projectPoints(transformed_points, rvec, tvec, cam_matrix, dist_coeffs)[0]

    mask = (ss_reproj[..., 0] >= 0) & (ss_reproj[..., 0] < im_w) & (ss_reproj[..., 1] >= 0) & (ss_reproj[..., 1] < im_h)
    ss_reproj = ss_reproj[mask]

    mask = mask[:, 0]

    inds = np.asarray(np.floor(ss_reproj), dtype=np.int32)

    scatter_im = np.zeros((im_h, im_w, 3))
    scatter_im[inds[..., 1], inds[..., 0]] = points_rgb[mask]

    plt.imsave(f"{save_dir}/{str(i).zfill(4)}.png", scatter_im * 0.5 + rgb_q * 0.5)


inp_idxs = list(zip(range(len(eimg_inds)), eimg_inds))
parallel_map(projection_image, inp_idxs, show_pbar=True, desc="final imgs")


