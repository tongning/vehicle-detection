import sys
sys.path.append('../darkflow')
from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from StereoDepth import *
import visualize2d

class NetworkModel:
    def __init__(self):

        self.vd_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        os.chdir(self.vd_directory)
        os.chdir("darkflow")
        options = {"model": os.path.join(self.vd_directory, "network/cfg/kitti.cfg"),
                   "load": -1,
                   "threshold": 0.01,
                   "gpu": 0.8}

        self.tfnet = TFNet(options)

    def PredictFrame(self, sequence_name, image_name):
        #directory_l = os.path.join(self.vd_directory, "data/KITTI-tracking/training/image_02/", sequence_name)
        #directory_r = os.path.join(self.vd_directory, "data/KITTI-tracking/training/image_03/", sequence_name)
        if type(image_name) == int:
            image_name = str(image_name).zfill(6) + '.png'
        image_path_l = os.path.join(self.vd_directory, "data/KITTI-tracking/training/image_02/", sequence_name, image_name)
        image_path_r = os.path.join(self.vd_directory, "data/KITTI-tracking/training/image_03/", sequence_name, image_name)

        bgr_image = cv2.imread(image_path_l)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        yolo_prediction = self.tfnet.return_predict(rgb_image)
        stereoPrediction = Convert3D(image_path_l, image_path_r, yolo_prediction)

        frame_data = {'tracked_objects': [], 'image_l': rgb_image, 'image_depth': stereoPrediction.depth_img, 'point_cloud': stereoPrediction.point_cloud}
        for predicted_box, predicted_3d_position in zip(yolo_prediction, stereoPrediction.positions_3D):
            tracked_object = {}
            tracked_object['bbox'] = {'left': predicted_box['topleft']['x'], 'top': predicted_box['topleft']['y'], 'right': predicted_box['bottomright']['x'], 'bottom': predicted_box['bottomright']['y']}
            tracked_object['confidence'] = predicted_box['confidence']
            tracked_object['type'] = predicted_box['label']
            tracked_object['3dbbox_loc'] = predicted_3d_position
            frame_data['tracked_objects'].append(tracked_object)

        return frame_data


    def PredictSequence(self, sequence_name = '0010', visualize=False):
        directory_l = os.path.join(self.vd_directory, "data/KITTI-tracking/training/image_02/", sequence_name)
        directory_r = os.path.join(self.vd_directory, "data/KITTI-tracking/training/image_03/", sequence_name)

        out_directory = os.path.join(self.vd_directory, 'eval', sequence_name, 'predictions')
        os.makedirs(out_directory, exist_ok=True)
        # Iterate over images

        _, ax = plt.subplots(figsize=(20, 10))
        im = None

        for i, filename in enumerate(sorted(os.listdir(directory_l))):
            if filename.endswith('.png'):
                print("Sequence: ", sequence_name, "  ", i, "/", len(os.listdir(directory_l)), end='\r', flush=True)
                frame_data = self.PredictFrame(sequence_name, filename)
                out_file_name = os.path.join(out_directory, os.path.splitext(filename)[0])

                if visualize:
                    img = visualize2d.Draw2DBoxes(frame_data)
                    if not im:
                        im = ax.imshow(img)
                    else:
                        im.set_data(img)
                    plt.pause(0.01)
                    plt.draw()

                with open(out_file_name, 'wb+') as out_file:
                    pickle.dump(frame_data, out_file, pickle.HIGHEST_PROTOCOL)



"""
# DEPRECATED #
def OutPrediction(prediction, frame_number, out_directory, image_l_path, image_r_path):
    # Save prediction as a file.
    os.makedirs(out_directory, exist_ok=True)
    out_file_name = os.path.join(out_directory, str(frame_number).zfill(4))
    stereoPrediction = Convert3D(image_l_path, image_r_path, prediction)
    with open(out_file_name, 'wb+') as out_file:
        frame = []
        for predicted_box, predicted_3d_position in zip(prediction, stereoPrediction.positions_3D):
            tracked_object = {}
            tracked_object['bbox'] = {'left': predicted_box['topleft']['x'], 'top': predicted_box['topleft']['y'], 'right': predicted_box['bottomright']['x'], 'bottom': predicted_box['bottomright']['y']}
            tracked_object['confidence'] = predicted_box['confidence']
            tracked_object['type'] = predicted_box['label']
            tracked_object['3dbbox_loc'] = predicted_3d_position
            frame.append(tracked_object)

        pickle.dump(frame, out_file, pickle.HIGHEST_PROTOCOL)
"""
