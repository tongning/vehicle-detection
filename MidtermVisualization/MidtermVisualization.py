from StereoDepth import Convert3D
import numpy as np
import os
import matplotlib.pyplot as plt
from onlinekalman import MultiOnlineKalman
from mpl_toolkits.mplot3d import Axes3D
import cv2
import open3d as o3d

#
# To use: Just run python 'MidtermVisualization.py'
#
#
# To change camera angle: Run 'python ChangeVisualizationViewpoint.py', move the camera
# around until it's where you want, then press q. Then run 'MidtermVisualization.py'
#
#




def bounding_box(position_3d, alpha=0, bbox_size = (10, 10, 10)):
    # pos is a list of xyz, e.g. [0.45 3.10 5.0]
    points = [[-bbox_size[0], -bbox_size[1], -bbox_size[2]],
            [bbox_size[0], -bbox_size[1], -bbox_size[2]],
            [-bbox_size[0], bbox_size[1], -bbox_size[2]],
            [bbox_size[0], bbox_size[1], -bbox_size[2]],
            [-bbox_size[0], -bbox_size[1], bbox_size[2]],
            [bbox_size[0], -bbox_size[1], bbox_size[2]],
            [-bbox_size[0], bbox_size[1], bbox_size[2]],
            [bbox_size[0], bbox_size[1], bbox_size[2]]]
    points = (np.int16(points) + 10*np.int16(np.array(pos))).tolist()
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[0, 1, 0] for i in range(len(lines))] # set bounding box color to green
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set



current_directory = os.getcwd()
info = np.load(current_directory + '/../0010-left-info.pyc', allow_pickle=True)
info_labels = np.load(current_directory + '/../dims_labels.npy')
print(info_labels)
#directory_l = current_directory + '/../data_tracking_image_2/training/image_02/0010/'
#directory_r = current_directory + '/../data_tracking_image_2/training/image_03/0010/'
directory_l = current_directory + '/../kitti/0010-left/'
directory_r = current_directory + '/../kitti/0010-right/'

vis = o3d.visualization.Visualizer()
pcd = o3d.geometry.PointCloud()
vis.create_window()

opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0]) # set background to black
opt.point_size = 5
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
        pcd.points = o3d.utility.Vector3dVector(np.int16(frame.point_cloud * 10))
        # Use StereoDepth Conversions---------------------------------

        if i == 0:
            # Add point cloud at first step, otherwise update
            vis.add_geometry(pcd)


        # add bounding boxes-------------------------------------------
        for pos in kalman.take_multiple_observations(frame.positions_3D): #frame.positions_3D is a list of positions (multiple if we detect more than one car in the same frame)
            # pos is a list of xyz, e.g. [0.45 3.10 5.0]
            bbox = bounding_box(pos)
            vis.add_geometry(bbox) # add bounding box to visualizer
            bounding_boxes.append(bbox) # add it to line_sets so we can clear it at the next iteration
        # add bounding boxes-------------------------------------------


        # Change Camera Position --------------------------------------
        ctr = vis.get_view_control() # load viewpoint
        param = o3d.io.read_pinhole_camera_parameters(current_directory + '/viewpoint.json')
        ctr.convert_from_pinhole_camera_parameters(param)
        # Change Camera Position --------------------------------------


        vis.poll_events()
        vis.update_renderer()
        vis.update_geometry()

        print(i)

        # Draw 2D image with 2D bounding boxes for debugging -------------------
        left_image = cv2.imread(filename_l)
        for bound in info[i]:
            if bound[4] == 'car' and float(bound[5]) > 0.9:
                cv2.rectangle(left_image, (int(bound[2]), int(bound[0])), (int(bound[3]), int(bound[1])), (255, 0, 0), 2)
        # Ground truth 2D bounding box.
        for label in info_labels[i]:
            cv2.rectangle(left_image, (int(label[3]), int(label[1])), (int(label[4]), int(label[2])), (0, 255, 0), 2)
        #plt.imshow(left_image, vmin=-1, vmax = 50)
        im.set_data(left_image)
        plt.pause(0.1)
        plt.draw()
        # Draw 2D image with 2D bounding boxes for debugging -------------------
