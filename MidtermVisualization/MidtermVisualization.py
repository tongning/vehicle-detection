from StereoDepth import Convert3D
import numpy as np
import os
import matplotlib.pyplot as plt
from onlinekalman import MultiOnlineKalman
from mpl_toolkits.mplot3d import Axes3D
import cv2
import open3d as o3d
import time

#
# To use: Just run python 'MidtermVisualization.py'
#
#
# To change camera angle: Run 'python ChangeVisualizationViewpoint.py', move the camera
# around until it's where you want, then press q. Then run 'MidtermVisualization.py'
#
#

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


SCALE_FACTOR = 100


current_directory = os.getcwd()
info = np.load(current_directory + '/../0010-left-info.pyc', allow_pickle=True)
info_labels = np.load(current_directory + '/../dims_labels.npy', allow_pickle=True)
directory_l = current_directory + '/../data/KITTI-tracking/training/image_02/0010/'
directory_r = current_directory + '/../data/KITTI-tracking/training/image_03/0010/'

vis = o3d.visualization.Visualizer()
pcd = o3d.geometry.PointCloud()
vis.create_window()

opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0]) # set background to black
opt.point_size = 1
opt.line_width = 100    # doesn't seem to be working


max_files = 100 # run on first 100 frames
bounding_boxes = [] # 3D bounding boxes


kalman = MultiOnlineKalman()

fig, ax = plt.subplots()
im = ax.imshow(cv2.imread(os.path.join(directory_l, '000000.png')))


for i, filename in enumerate(sorted(os.listdir(directory_l))):
    if i > max_files:
        break

    if filename.endswith('.png'):
        filename_l = os.path.join(directory_l, filename)
        filename_r = os.path.join(directory_r, filename)

        # clear old bounding boxes --------
        for bbox in bounding_boxes:
            vis.remove_geometry(bbox)
        # clear old bounding boxes --------


        # Use StereoDepth Conversions---------------------------------
        frame = Convert3D(filename_l, filename_r, info[i])
        # add point cloud
        scaled_pc = np.clip(frame.point_cloud * SCALE_FACTOR, -10000, 10000)
        pcd.points = o3d.utility.Vector3dVector(np.int16(scaled_pc))
        # Use StereoDepth Conversions---------------------------------

        if i == 0:
            # Add point cloud at first step, otherwise update
            vis.add_geometry(pcd)
            pass

            # add bounding boxes-------------------------------------------
        for pos in frame.positions_3D:  # frame.positions_3D is a list of positions (multiple if we detect more than one car in the same frame)
            # pos is a list of xyz, e.g. [0.45 3.10 5.0]
            color = [1, 0, 0]
            bbox = bounding_box(pos, color)
            vis.add_geometry(bbox)  # add bounding box to visualizer
            bounding_boxes.append(bbox)  # add it to line_sets so we can clear it at the next iteration
        # add bounding boxes-------------------------------------------


        # add bounding boxes-------------------------------------------
        for pos in kalman.take_multiple_observations(frame.positions_3D): #frame.positions_3D is a list of positions (multiple if we detect more than one car in the same frame)
            # pos is a list of xyz, e.g. [0.45 3.10 5.0]
            color = [0, 0, 1]
            bbox = bounding_box(pos, color)
            vis.add_geometry(bbox) # add bounding box to visualizer
            bounding_boxes.append(bbox) # add it to line_sets so we can clear it at the next iteration
        # add bounding boxes-------------------------------------------

        # Add 3D bounding box ground truth labels
        for label in info_labels[i]:
            # [0] alpha, [5] 3d_height, [6] 3d_width, [7] 3d_length, [8] x, [9] y, [10] z
            # TODO: Is alpha being used inside bounding_box ?
            pos = [label[8], label[9], label[10]]
            color = [0, 1, 0]
            #bbox = bounding_box(pos, color, label[0])
            bbox = bounding_box(pos, color)
            vis.add_geometry(bbox)
            bounding_boxes.append(bbox)



        # Change Camera Position --------------------------------------
        ctr = vis.get_view_control() # load viewpoint
        param = o3d.io.read_pinhole_camera_parameters(current_directory + '/viewpoint.json')
        ctr.convert_from_pinhole_camera_parameters(param)
        # Change Camera Position --------------------------------------


        vis.poll_events()
        vis.update_renderer()
        vis.update_geometry()

        #capture_image(vis)


        # Draw 2D image with 2D bounding boxes for debugging -------------------
        left_image = cv2.imread(filename_l)
        pred_dims = info[i]
        for bound in pred_dims:
            if bound[4] == 'car' and float(bound[5]) > 0.9:
                cv2.rectangle(left_image, (int(bound[2]), int(bound[0])), (int(bound[3]), int(bound[1])), (0, 0, 255), 2)

        # Draw 2D image with 2D bounding boxes for debugging -------------------
        actual_dims = info_labels[i]
        for label in actual_dims:
            cv2.rectangle(left_image, (int(label[3]), int(label[1])), (int(label[4]), int(label[2])), (0, 255, 0), 2)


        #plt.imshow(left_image, vmin=-1, vmax = 50)
        im.set_data(left_image)
        plt.pause(0.1)
        plt.draw()
