import numpy as np
import sys

from darkflow.net.build import TFNet
import cv2


# Ckpt kitti-43750
# Tuned with low learning rate

# Ckpt kitti-52250
# Tuned with 1e-5 learning rate

# Retrained on original

options = {"model": "/home/eric/vehicle-detection/network/rgb-multi-class-1e4/kitti.cfg",
            "load": -1,
            #"load": "/home/eric/vehicle-detection/network/weights/yolo.weights",
            "batch": 8,
            "epoch": 500,
            "gpu": 0.9,
            "train": True,
            "trainer": "adam",
            "lr": 1e-4,
            "annotation": "/home/eric/vehicle-detection/data/KITTI-detection-angles/Annotations/",
            "dataset": "/home/eric/vehicle-detection/data/KITTI-detection-depth/image_2/"}
            #"annotation": "/home/eric/vehicle-detection/data/KITTI-Detection-VOC/Annotations",
            #"dataset": "/home/eric/vehicle-detection/data/KITTI-Detection-VOC/JPEGImages"}


#os.chdir("../darkflow")

tfnet = TFNet(options)

tfnet.train()
