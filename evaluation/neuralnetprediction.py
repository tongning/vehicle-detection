import sys
sys.path.append('../darkflow')
from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from StereoDepth import *

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


class NetworkModel:
    def __init__(self):
        options = {"model": "/home/eric/vehicle-detection/network/rgb-multi-class-1e4/kitti.cfg",
                   "load": -1,
                   "threshold": 0.01,
                   "gpu": 0.8}

        self.original_directory = os.getcwd()
        os.chdir("../darkflow")
        self.tfnet = TFNet(options)

    def PredictSequence(self, sequence_name = '0010'):
        directory_l = os.path.join("/home/eric/vehicle-detection/data/KITTI-tracking/training/image_02/", sequence_name)
        directory_r = os.path.join("/home/eric/vehicle-detection/data/KITTI-tracking/training/image_03/", sequence_name)
        for i, filename in enumerate(sorted(os.listdir(directory_l))):
            if filename.endswith('.png'):
                filename_l = os.path.join(directory_l, filename)
                filename_r = os.path.join(directory_r, filename)
                frame = cv2.imread(filename_l)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prediction = self.tfnet.return_predict(frame)

                OutPrediction(prediction, os.path.splitext(filename)[0], os.path.join(self.original_directory, 'eval', sequence_name, 'predictions'), filename_l, filename_r)
