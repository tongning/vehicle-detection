################################################################################
#
# Plays a video showing 3D point cloud, predictions, and ground truth boxes.
# The brightness of the red predictions is proportional to the model's
# confidence.
#
# Usage:
# python visualization3d.py 0010
# python visualization3d.py 0011 0014 0011
#
################################################################################






import sys
import os
vd_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(vd_directory)
sys.path.append(os.path.join(vd_directory, 'evaluation'))
from StereoDepth import Convert3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import open3d as o3d
import pickle

SCALE_FACTOR = 100

def loadFrameData(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def PlaySequence(sequence_name):
    vd_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    groundtruth_path = os.path.join(vd_directory, 'eval', sequence_name, 'groundtruth')
    prediction_path = os.path.join(vd_directory, 'eval', sequence_name, 'predictions')




    vis = o3d.visualization.Visualizer()
    pcd = o3d.geometry.PointCloud()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0]) # set background to black
    opt.point_size = 1
    opt.line_width = 100    # doesn't seem to be working

    bounding_boxes = [] # 3D bounding boxes

    i = 0

    for file_name in sorted(os.listdir(prediction_path)):
        if file_name == '.DS_Store':
            continue
        prediction_file_path = os.path.join(prediction_path, file_name)
        groundtruth_file_path = os.path.join(groundtruth_path, file_name)

        prediction_frame_data = loadFrameData(prediction_file_path)
        groundtruth_frame_data = loadFrameData(groundtruth_file_path)

        pc = prediction_frame_data['point_cloud']
        scaled_pc = np.clip(pc * SCALE_FACTOR, -10000, 10000)
        pcd.points = o3d.utility.Vector3dVector(np.int16(scaled_pc))

        # clear old bounding boxes --------
        for bbox in bounding_boxes:
            vis.remove_geometry(bbox)
        # clear old bounding boxes --------

        if i == 0:
            # Add point cloud at first step, otherwise update
            vis.add_geometry(pcd)
            pass

        # add predicted bounding boxes-------------------------------------------
        for tracked_object in prediction_frame_data['tracked_objects']:  # frame.positions_3D is a list of positions (multiple if we detect more than one car in the same frame)
            # pos is a list of xyz, e.g. [0.45 3.10 5.0]
            pos = tracked_object['3dbbox_loc']
            color = [tracked_object['confidence'], 0, 0]
            bbox = bounding_box(pos, color)
            vis.add_geometry(bbox)  # add bounding box to visualizer
            bounding_boxes.append(bbox)  # add it to line_sets so we can clear it at the next iteration


        # Add 3D bounding box ground truth labels
        for tracked_object in groundtruth_frame_data['tracked_objects']:
            # [0] alpha, [5] 3d_height, [6] 3d_width, [7] 3d_length, [8] x, [9] y, [10] z
            # TODO: Is alpha being used inside bounding_box ?
            #pos = [label[8], label[9], label[10]]
            pos = tracked_object['3dbbox_loc']
            color = [0, 1, 0]
            #bbox = bounding_box(pos, color, label[0])
            bbox = bounding_box(pos, color)
            vis.add_geometry(bbox)
            bounding_boxes.append(bbox)


        # Change Camera Position --------------------------------------
        ctr = vis.get_view_control() # load viewpoint
        param = o3d.io.read_pinhole_camera_parameters('viewpoint.json')
        ctr.convert_from_pinhole_camera_parameters(param)
        # Change Camera Position --------------------------------------


        vis.poll_events()
        vis.update_renderer()
        vis.update_geometry()
        i += 1




def capture_image(vis):
    image = vis.capture_screen_float_buffer()
    plt.imshow(np.asarray(image))
    #plt.show()
    return False


def bounding_box(pos, color, alpha=0, bbox_size=(1, 1, 1)):
    # pos is a list of xyz, e.g. [0.45 3.10 5.0]
    points = [[-bbox_size[0], -bbox_size[1], -bbox_size[2]],
            [bbox_size[0], -bbox_size[1], -bbox_size[2]],
            [-bbox_size[0], bbox_size[1], -bbox_size[2]],
            [bbox_size[0], bbox_size[1], -bbox_size[2]],
            [-bbox_size[0], -bbox_size[1], bbox_size[2]],
            [bbox_size[0], -bbox_size[1], bbox_size[2]],
            [-bbox_size[0], bbox_size[1], bbox_size[2]],
            [bbox_size[0], bbox_size[1], bbox_size[2]]]
    points = np.int16(SCALE_FACTOR * (points + np.array(pos))).tolist()
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Set color of bounding box. Green for ground truth, blue for prediction w/ filter, red is prediction without filter
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def main(argv):
    for sequence in argv[1:]:
        PlaySequence(sequence)

if __name__ == '__main__':
    main(sys.argv)
