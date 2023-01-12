import sys
sys.path.append(".")

import argparse

import numpy as np
from colmap_find_scale.read_write_model import read_points3D_binary
from exviz.util.camera_pose_visualizer import CameraPoseVisualizer
import open3d

def open3d_reject_outliers(points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=20,
                                              std_ratio=2.0)

    return np.asarray(pcd.points)

def read_3dpoints(path):
    points3d = read_points3D_binary(path)
    points = []
    for k, v in points3d.items():
        points.append(v.xyz)
    
    points = np.stack(points)
    return points


def plot_framewise(col_path, ecam_path, pnts_path):
    
    e_mtxs = np.load(ecam_path)
    c_mtxs = np.load(col_path)
    max_frame_length = 200 
    gap_size = len(c_mtxs)//max_frame_length

    e_coords = e_mtxs[:,:3,3]
    coords = c_mtxs[:,:3,3]

    c_mtxs = c_mtxs[::gap_size]
    e_mtxs = e_mtxs[::gap_size]

    graph_bounds = []

    points = read_3dpoints(pnts_path)
    points = open3d_reject_outliers(points)
    for i in range(3):
        graph_bounds.append([min(coords[:,i].min(), e_coords[:,i].min(), points[:,i].min()),
                             max(coords[:,i].max(), e_coords[:,i].max(), points[:,i].max())])

    visualizer = CameraPoseVisualizer(graph_bounds[0], graph_bounds[1], graph_bounds[2])

    pyramid_scale = 0.25
    for _, extrinsic in enumerate(c_mtxs):
        visualizer.extrinsic2pyramid(extrinsic, "r", pyramid_scale)

    for _, extrinsic in enumerate(e_mtxs):
        visualizer.extrinsic2pyramid(extrinsic, "g", pyramid_scale)
    
    
    gap = len(points)//100
    # points = points[::gap]   # for showing fewer points
    x, y, z = [points[:,i] for i in range(3)]
    visualizer.ax.scatter(x,y,z, s=0.2)


    visualizer.colorbar(max_frame_length)
    visualizer.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize the color and event camera trajectories")
    parser.add_argument("-c", "--col_ext_path", help="path to color camera extrinsics")
    parser.add_argument("-e", "--ecam_ext_path", help="path to event camera extrinsics")
    parser.add_argument("-p", "--pnts_3d_path", help="path to colmap 3d points of the scene")
    args = parser.parse_args()

    plot_framewise(args.col_ext_path, args.ecam_ext_path, args.pnts_3d_path)
