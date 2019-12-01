################################################################################
#
# Plays a video showing the 2D image space predictions and ground truth boxes.
# The brightness of the red predictions is proportional to the model's
# confidence.
#
# Usage:
# python visualize2d 0010
# python visualize2d 0011 0014 0011
#
################################################################################


import matplotlib.pyplot as plt
import os
import sys
import pickle
import cv2

def loadFrameData(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def Draw2DBoxes(prediction=None, groundtruth=None, image=None):
    if prediction:
        image = prediction['image_depth']

    if groundtruth:
        for tracked_object in groundtruth['tracked_objects']:
            bbox = tracked_object['bbox']
            image = cv2.rectangle(image, (bbox['left'], bbox['top']), (bbox['right'], bbox['bottom']), (0, 255, 0), 2)
    if prediction:
        for tracked_object in prediction['tracked_objects']:
            bbox = tracked_object['bbox']
            image = cv2.rectangle(image, (bbox['left'], bbox['top']), (bbox['right'], bbox['bottom']), (int(tracked_object['confidence']*255), 0, 0), 2)

    return image

def PlaySequence(sequence_name):
    vd_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    groundtruth_path = os.path.join(vd_directory, 'eval', sequence_name, 'groundtruth')
    prediction_path = os.path.join(vd_directory, 'eval', sequence_name, 'predictions')

    _, ax = plt.subplots(figsize=(20, 10))
    im = None

    for file_name in sorted(os.listdir(prediction_path)):
        if file_name == '.DS_Store':
            continue
        prediction_file_path = os.path.join(prediction_path, file_name)
        groundtruth_file_path = os.path.join(groundtruth_path, file_name)

        img = Draw2DBoxes(loadFrameData(prediction_file_path), loadFrameData(groundtruth_file_path))

        if not im:
            im = ax.imshow(img)
        else:
            im.set_data(img)
        plt.pause(0.01)
        plt.draw()


def main(argv):
    for sequence in argv[1:]:
        PlaySequence(sequence)

if __name__ == '__main__':
    main(sys.argv)
