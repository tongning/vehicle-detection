import pickle
import numpy as np
import os

def loadFrameData(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def TPFP2D(predictions, groundtruth, iou_threshold=0.5):
    # Labels each predicted 2D bounding box for a given image with
    # True Positive or False Positive.
    # Sort predicted objects by descending confidence, then match them to the
    # best ground truth box. If the iou is greater than the iou_threshold,
    # label as a true positive, else as a false positive. We ensure
    # that we don't 'reuse' ground truth boxes; if we have a double prediction
    # on a car, only one will be a true positive. The other will be a false
    # positive.
    results = []
    used_gt_objects = set()
    gt_objects = set(range(len(groundtruth['tracked_objects'])))
    for tracked_object in sorted(predictions['tracked_objects'], key=lambda x: x['confidence'], reverse=True):

        if not gt_objects:
            results.append((tracked_object['confidence'], 'FP', 0))
            continue
        min_distance, gt_object = max([(IOU(tracked_object, groundtruth['tracked_objects'][gt_object]), gt_object) for gt_object in gt_objects])

        if min_distance >= iou_threshold:
            gt_objects.remove(gt_object)
            #used_gt_objects.add(gt_index)
            results.append((tracked_object['confidence'], 'TP', min_distance))
        else:
            results.append((tracked_object['confidence'], 'FP', min_distance))
    return results

def TPFP3D(predictions, groundtruth, distance_threshold=5):
    # Labels each predicted 3D bounding box for a given image with
    # True Positive or False Positive.
    # Sort predicted objects by descending confidence, then match them to the
    # best ground truth location. If the distance is less than the distance_threshold,
    # label as a true positive, else as a false positive. We ensure
    # that we don't 'reuse' ground truth locations; if we have a double prediction
    # on a car, only one will be a true positive. The other will be a false
    # positive.

    results = []
    #used_gt_objects = set()
    gt_objects = set(range(len(groundtruth['tracked_objects'])))
    for tracked_object in sorted(predictions['tracked_objects'], key=lambda x: x['confidence'], reverse=True):

        if not gt_objects:
            results.append((tracked_object['confidence'], 'FP', 0))
            continue
        min_distance, gt_object = min([(EuclideanDistance(tracked_object, groundtruth['tracked_objects'][gt_object]), gt_object) for gt_object in gt_objects])

        if min_distance <= distance_threshold:
            gt_objects.remove(gt_object)
            #used_gt_objects.add(gt_index)
            results.append((tracked_object['confidence'], 'TP', min_distance))
        else:
            results.append((tracked_object['confidence'], 'FP', min_distance))
    return results

def EuclideanDistance(groundtruth_object, predicted_object):
    return np.linalg.norm(np.array(groundtruth_object['3dbbox_loc']) - np.array(predicted_object['3dbbox_loc']))

def IOU(A, B):
    boxA = A['bbox']
    boxB = B['bbox']
    # from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    areaA = (boxA['right'] - boxA['left']) * (boxA['bottom'] - boxA['top'])
    areaB = (boxB['right'] - boxB['left']) * (boxB['bottom'] - boxB['top'])

    left = max(boxA['left'], boxB['left'])
    right = min(boxA['right'], boxB['right'])
    top = max(boxA['top'], boxB['top'])
    bottom = min(boxA['bottom'], boxB['bottom'])

    interArea = max(0, right-left) * max(0, bottom - top)
    iou = interArea / float(areaA + areaB - interArea)

    # return the intersection over union value
    return iou


def PR(type='2D', threshold=0.5):
        units = str(threshold*100) + ' %' if type == '2D' else str(threshold) + ' m'

        TPFP_table = []
        total_ground_truth = 0

        for sequence_name in os.listdir('eval'):
            if sequence_name == '.DS_Store':
                continue

            print ("Sequence: ", sequence_name, " Type: ", type, " Threshold: ", units)
            for i, filename in enumerate(sorted(os.listdir(os.path.join('eval',sequence_name,'predictions')))):
                prediction = loadFrameData(os.path.join('eval', sequence_name, 'predictions', filename))
                groundtruth = loadFrameData(os.path.join('eval', sequence_name, 'groundtruth', filename))
                total_ground_truth += len(groundtruth['tracked_objects'])
                if type == '2D':
                    TPFP_table += TPFP2D(prediction, groundtruth, threshold)
                else:
                    TPFP_table += TPFP3D(prediction, groundtruth, threshold)

        TPFP_table.sort(reverse=True) # Sort by descending confidence
        TP = 0 # True positive counts
        FP = 0 # False positive counts
        precision = []
        recall = []
        for pred in TPFP_table:
            if pred[1] == 'TP':
                TP += 1
            else:
                FP += 1
            p = TP / (TP + FP)
            r = TP / total_ground_truth
            precision.append(p)
            recall.append(r)

        return precision, recall
