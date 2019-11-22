import cv2
import matplotlib.pyplot as plt
import pickle
from iou import *
import numpy as np
import os
from neuralnetprediction import *
from groundtruth import *

def loadFrameData(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def visualize2d(image, prediction=[], groundtruth=[]):
    for tracked_object in prediction:
        bbox = tracked_object['bbox']
        image = cv2.rectangle(image, (bbox['left'], bbox['top']), (bbox['right'], bbox['bottom']), (255, 0, 0), 2)

    for tracked_object in groundtruth:
        bbox = tracked_object['bbox']
        image = cv2.rectangle(image, (bbox['left'], bbox['top']), (bbox['right'], bbox['bottom']), (0, 255, 0), 2)

    plt.imshow(image)
    plt.show()


def TPFP(predictions, groundtruth, iou_threshold=0.5):
    # For each prediction in the frame, mark it as a true positive or a false positive.
    results = []

    matchings = []
    for g, groundtruth_object in enumerate(groundtruth):
        matchings += [(IOU(groundtruth_object, predicted_object), g, p) for p, predicted_object in enumerate(predictions)]

    matchings.sort(reverse=True)
    used = set()
    for matching in matchings:
        iou, g, p = matching
        if iou <= iou_threshold:
            break
        if ('g', g) not in used and ('p', p) not in used:
            used.add(('g', g))
            used.add(('p', p))

    for p, prediction in enumerate(predictions):
        if ('p', p) in used:
            results.append(('TP', prediction['confidence']))
        else:
            results.append(('FP', prediction['confidence']))

    return results



def EuclideanDistance(groundtruth_object, predicted_object):
    return np.linalg.norm(np.array(groundtruth_object['3dbbox_loc']) - np.array(predicted_object['3dbbox_loc']))


def TPFP3D(predictions, groundtruth, distance_threshold=5):
    # For each prediction in the frame, mark it as a true positive or a false positive.

    results = []

    matchings = []
    for g, groundtruth_object in enumerate(groundtruth):
        matchings += [(EuclideanDistance(groundtruth_object, predicted_object), g, p) for p, predicted_object in enumerate(predictions)]

    matchings.sort()
    used = set()
    for matching in matchings:
        iou, g, p = matching
        if iou > distance_threshold:
            break
        if ('g', g) not in used and ('p', p) not in used:
            used.add(('g', g))
            used.add(('p', p))

    for p, prediction in enumerate(predictions):
        if ('p', p) in used:
            results.append(('TP', prediction['confidence']))
        else:
            results.append(('FP', prediction['confidence']))

    return results





def main(argv):
    directory = os.getcwd()
    if len(argv) > 1 and argv[1] == 'generate':
        if len(argv) > 2:
            num_seq = int(argv[2]) + 1
        else:
            num_seq = 21

        model = NetworkModel()
        print("Running Model")

        for seq_num in range(num_seq):
            sequence_name = str(seq_num).zfill(4)
            print ("Sequence: ", sequence_name)
            model.PredictSequence(sequence_name)
            OutGroundTruthSequence(sequence_name)

    TPFP_table = []
    total_ground_truth = 0

    os.chdir(directory)
    print (os.listdir('eval'))
    for sequence_name in os.listdir('eval'):
        #sequence_name = str(seq_num).zfill(4)
        print ("Sequence: ", sequence_name)
        for i, filename in enumerate(sorted(os.listdir(os.path.join('eval',sequence_name,'predictions')))):
            prediction = loadFrameData(os.path.join('eval', sequence_name, 'predictions', filename))
            groundtruth = loadFrameData(os.path.join('eval', sequence_name, 'groundtruth', filename))
            total_ground_truth += len(groundtruth)
            TPFP_table += TPFP3D(prediction, groundtruth, 5)

    # Sort by descending confidence
    TPFP_table.sort(key=lambda x: x[1], reverse=True)
    TP = 0
    FP = 0

    precision = []
    recall = []

    for pred in TPFP_table:
        if pred[0] == 'TP':
            TP += 1
        else:
            FP += 1

        p = TP / float(TP + FP)
        r = TP / total_ground_truth


        precision.append(p)
        recall.append(r)

    axes = plt.gca()
    axes.set_xlim([0, 1.2])
    axes.set_ylim([0, 1.2])
    plt.plot(recall, precision, )
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
