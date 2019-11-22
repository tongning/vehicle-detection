import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d


class Convert3D:
    def __init__(self, img_l, img_r, prediction = []):

        if isinstance(img_l, str):
            self.img_l = cv2.imread(img_l)
            self.img_r = cv2.imread(img_r)
        else:
            self.img_l = img_l
            self.img_r = img_r

        self.prediction = prediction

        self.depth_img = self._CalculateStereoDisparityFast()
        #self.depth_img = self._CalculateStereoDisparity()
        self.xyz_img = -1.2*self._XYZImageFromDisparity()
        self.point_cloud = self.PointCloud()
        self.positions_3D = self.Positions3D()


    def _CalculateStereoDisparity(self):
        # SGBM Parameters
        window_size = 3

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=160, # must be divisible by 16?
            blockSize=5,
            P1=8 * 3 * window_size ** 2,
            P2 = 32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        # Filter Params
        lmbda = 80000
        sigma = 1.2
        visual_multiplier = 1.0

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        displ = left_matcher.compute(self.img_l, self.img_r).astype(np.float32)/16.0
        dispr = right_matcher.compute(self.img_r, self.img_l).astype(np.float32)/16.0
        filteredImg = wls_filter.filter(displ, self.img_l, None, dispr)

        return filteredImg

    def _CalculateStereoDisparityFast(self):
        # SGBM Parameters
        window_size = 5

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=160, # must be divisible by 16?
            blockSize=5,
            P1=8 * 3 * window_size ** 2,
            P2 = 32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        displ = left_matcher.compute(self.img_l, self.img_r).astype(np.float32) / 16.0
        return displ

    def _XYZImageFromDisparity(self):
        f = 7.215377000000e+02
        cx = 6.095593000000e+02
        cx1 = 526.242457
        cy = 1.728540000000e+02
        Tx = -120.00

        Q = np.float32([[   1.,            0.,            0.,         -614.37893072],
            [   0.,            1.,            0.,         -162.12583194],
            [   0.,            0.,            0.,          680.05186262],
            [   0.,            0.,           -1.87703644,    0.,        ]])

        points = cv2.reprojectImageTo3D(self.depth_img, Q)
        return points

    def PointCloud(self):
        mask = self.depth_img > self.depth_img.min()
        return self.xyz_img[mask]

    def Positions3D(self):
        positions = []

        for predicted_box in self.prediction:
            xyz_car_img = self.xyz_img[predicted_box['topleft']['y']:predicted_box['bottomright']['y'], predicted_box['topleft']['x']:predicted_box['bottomright']['x']]
            h, w, _ = xyz_car_img.shape
            # Crop
            xyz_car_img = xyz_car_img[int(h/4):int(3*h/4), int(w/4):int(3*w/4)]

            # Get point
            car_pos = np.median(xyz_car_img.reshape(-1, 3), axis=0)
            #car_pos[2] -= 2
            positions.append(car_pos)
        return positions
