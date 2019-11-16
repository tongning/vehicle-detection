import numpy as np
import sys

from darkflow.net.build import TFNet
import cv2



options = {"model": "/home/eric/vehicle-detection/network/cfg/kitti.cfg",
            "load": -1,#"/home/eric/vehicle-detection/network/weights/yolo.weights",
            "batch": 8,
            "epoch": 30,
            "gpu": 0.9,
            "train": True,
            "lr": 5e-8,
            "annotation": "/home/eric/vehicle-detection/data/KITTI-Detection-VOC/Annotations/",
            "dataset": "/home/eric/vehicle-detection/data/KITTI-Detection-VOC/JPEGImages/"}


#os.chdir("../darkflow")

tfnet = TFNet(options)

tfnet.train()
