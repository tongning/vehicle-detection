import numpy as np
import sys
sys.path.append("..")

from darkflow.darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo-kitti.cfg",
            "load": -1,#"weights/yolo.weights",
            "batch": 7,
            "epoch": 30,
            "gpu": 0.8,
            "train": True,
            "lr": 1e-7,
            "annotation": "../VOC2012/Annotations/",
            "dataset": "../VOC2012/JPEGImages/"}

tfnet = TFNet(options)

tfnet.train()
