import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
# Internal libraries
from onlinekalman import MultiOnlineKalman
from StereoDepth import Convert3D


def get_scaled_dims(image_x, image_y, x, z, depth, width):

    # TODO: Assuming range of values from -50 to 50.
    x += 50
    z += 50
    x *= (image_x / 100)
    z *= (image_y / 100)

    #return x,z
    width += 50
    depth += 50
    width *= (image_x / 100)
    depth *= (image_z / 100)

    bottom = x - (depth // 2)
    left = z - (width // 2)

    return bottom, left, depth, width


current_directory = os.getcwd()
pred_np = np.load(current_directory + '/../0010-left-info.pyc', allow_pickle=True)
label_np = np.load(current_directory + '/../dims_labels.npy')

directory_l = current_directory + '/../data_tracking_image_2/training/image_02/0010/'
directory_r = current_directory + '/../data_tracking_image_2/training/image_03/0010/'

pred1 = pred_np[0]
label1 = label_np[0]

print(pred1)
print(label1)

kalman = MultiOnlineKalman()
max_files = 1

#est_height = 200 # ?
#est_width = 100 # ?
est_depth = 50

fig, ax = plt.subplots(figsize=(124, 38)) # TODO: Do I have to set it up like this?

# Iterating over all the files we need to process, plotting each time we make a prediction.
for i, filename in enumerate(sorted(os.listdir(directory_l))):

    if i > max_files:
        break

    if filename.endswith('.png'):
        filename_l = os.path.join(directory_l, filename)
        filename_r = os.path.join(directory_r, filename)

        # TODO: Get the ACTUAL image_dims, this is a guess.
        image_x = 1240
        image_z = 375

        frame = Convert3D(filename_l, filename_r, pred_np[i])
        labels = label_np[i]

        # z is depth, y is height, x is width
        # # Positions of objects before we do kalman filter.
        for pos in frame.positions_3D:
            x = pos[0]
            z = pos[2]
            print("Pre-Kalman x %f z: %f" % (x, z))


        # # Positions of objects after we do kalman filter.
        for pos in kalman.take_multiple_observations(frame.positions_3D):
            x = pos[0]
            z = pos[2]
            print("Kalman x %f z: %f" % (x, z))


        # Positions of ground truth labels.
        for label in labels:
            x = label[8]
            z = label[9]
            width = label[6]
            depth = label[7]

            bottom = z - depth // 2
            left = x - width // 2

            print("Ground truth x: %f z: %f" % (x, z))

            bottom, left, depth, width = get_scaled_dims(image_x, image_z, x, z, width, depth)

            box = patches.Rectangle((bottom, left), depth, width, edgecolor='r')
            print("Printing box. Bot: %f Left: %f Depth: %f Width: %f" % (bottom, left, depth, width))
            ax.add_patch(box)

        plt.show()




