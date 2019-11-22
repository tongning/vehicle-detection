import sys
sys.path.append("../MidtermVisualization")
from StereoDepth import Convert3D
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image
import time

directory_l = "/home/eric/vehicle-detection/data/KITTI-detection-depth/image_2"
directory_r = "/home/eric/vehicle-detection/data/KITTI-detection-depth/image_3"
directory_out = "/home/eric/vehicle-detection/data/KITTI-detection-depth/depth"


plt.ion()
plt.show()
_, ax = plt.subplots(figsize=(20, 10))
im = ax.imshow(cv2.imread(os.path.join(directory_l, '000000.png')))


for i, filename in enumerate(sorted(os.listdir(directory_l))):

    if filename.endswith('.png'):
        filename_l = os.path.join(directory_l, filename)
        filename_r = os.path.join(directory_r, filename)

        image_l = cv2.imread(filename_l)
        image_r = cv2.imread(filename_r)

        image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)

        # Use StereoDepth Conversions---------------------------------
        frame = Convert3D(image_l, image_r)
        depth_img = np.clip(frame.depth_img * 2, 0, 255)
        #print(depth_img.shape)


        out_image = image_l
        out_image[:, :, 1] = depth_img
        filename_out = os.path.join(directory_out, filename)
        im = Image.fromarray(out_image)
        im.save(filename_out)
        print(i)
        time.sleep(0.5)
        #im.set_data(out_image)
        #plt.pause(0.61)
        #plt.draw()
        #plt.imshow(image)
        #plt.show()
