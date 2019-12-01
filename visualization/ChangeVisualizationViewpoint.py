# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/camera_trajectory.py
import sys
sys.path.append('..')
sys.path.append('../evaluation')
import numpy as np
import open3d as o3d
from StereoDepth import *
import os
import pickle

def loadFrameData(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    SCALE_FACTOR = 100

    prediction_file_path = '../eval/0011/predictions/000000'

    pcd = o3d.geometry.PointCloud()
    pc = loadFrameData(prediction_file_path)['point_cloud']
    scaled_pc = np.clip(pc * SCALE_FACTOR, -10000, 10000)
    pcd.points = o3d.utility.Vector3dVector(np.int16(scaled_pc))

    save_view_point(pcd, 'viewpoint.json')
    #load_view_point(pcd, current_directory + '/viewpoint.json')
