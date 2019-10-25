import matplotlib.pyplot as plt
import cv2
from StereoDepth import *
import os


directory_l = 'data_tracking_image_2/training/image_02/0010/'
directory_r = 'data_tracking_image_2/training/image_03/0010/'

for i, filename in enumerate(sorted(os.listdir(directory_l))):
    if filename.endswith('.png'):
        filename_l = os.path.join(directory_l, filename)
        filename_r = os.path.join(directory_r, filename)
        img_l = cv2.imread(filename_l)
        img_r = cv2.imread(filename_r)

        depth_image = CalculateStereoDisparityFast(img_l, img_r)
        #print(i)
        plt.imshow(depth_image, vmin=-1, vmax = 50)
        plt.pause(0.1)
        plt.draw()

plt.show()
