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
    gt_objects = set(range(len(groundtruth['tracked_objects'])))
    for tracked_object in sorted(predictions['tracked_objects'], key=lambda x: x['confidence'], reverse=True):

        if not gt_objects:
            results.append((tracked_object['confidence'], 'FP', 0, 'missing'))
            continue
        min_distance, gt_object = max([(IOU(tracked_object, groundtruth['tracked_objects'][gt_object]), gt_object) for gt_object in gt_objects])

        if min_distance >= iou_threshold:
            gt_objects.remove(gt_object)
            results.append((tracked_object['confidence'], 'TP', min_distance, groundtruth['tracked_objects'][gt_object]['difficulty'], int(tracked_object['type'].split('_')[1])))
        else:
            results.append((tracked_object['confidence'], 'FP', min_distance, groundtruth['tracked_objects'][gt_object]['difficulty']))
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
            results.append((tracked_object['confidence'], 'FP', 0, 'missing'))
            continue
        min_distance, gt_object = min([(EuclideanDistance(tracked_object, groundtruth['tracked_objects'][gt_object]), gt_object) for gt_object in gt_objects])

        if min_distance <= distance_threshold:
            gt_objects.remove(gt_object)
            #used_gt_objects.add(gt_index)
            results.append((tracked_object['confidence'], 'TP', min_distance, groundtruth['tracked_objects'][gt_object]['difficulty'], int(tracked_object['type'].split('_')[1])))
        else:
            results.append((tracked_object['confidence'], 'FP', min_distance, groundtruth['tracked_objects'][gt_object]['difficulty']))
    return results

def EuclideanDistance(groundtruth_object, predicted_object):
    # TODO: Create another version of this function for filtered bounding box location
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

# https://research.mapillary.com/img/publications/MonoDIS.pdf
# Calculate according to MonoDIS paper and KITTI readme
def MAP():
    pass


def PR(type='2D', threshold=0.5):
        units = str(threshold*100) + ' %' if type == '2D' else str(threshold) + ' m'

        TPFP_table = []
        easy_counts = 0
        medium_counts = 0
        hard_counts = 0

        for sequence_name in os.listdir('eval'):
            if sequence_name == '.DS_Store':
                continue

            print ("Sequence: ", sequence_name, " Type: ", type, " Threshold: ", units)
            for i, filename in enumerate(sorted(os.listdir(os.path.join('eval',sequence_name,'predictions')))):
                prediction = loadFrameData(os.path.join('eval', sequence_name, 'predictions', filename))
                groundtruth = loadFrameData(os.path.join('eval', sequence_name, 'groundtruth', filename))
                for gt_object in groundtruth['tracked_objects']:
                    if gt_object['difficulty'] == 'easy':
                        easy_counts += 1
                    elif gt_object['difficulty'] == 'medium':
                        medium_counts += 1
                    elif gt_object['difficulty'] == 'hard':
                        hard_counts += 1


                if type == '2D': # Image space precision-recall
                    TPFP_table += TPFP2D(prediction, groundtruth, threshold)
                else: # 3d position precision-recall
                    TPFP_table += TPFP3D(prediction, groundtruth, threshold)

        print(easy_counts)
        print(medium_counts)
        print(hard_counts)

        TPFP_table.sort(reverse=True) # Sort by descending confidence
        TP_easy = 0 # True positive counts
        FP_easy = 0 # False positive counts
        TP_medium = 0
        FP_medium = 0
        TP_hard = 0
        FP_hard = 0
        precision_easy = []
        precision_medium = []
        precision_hard = []
        recall_easy = []
        recall_medium = []
        recall_hard = []

        y_true = []
        y_pred = []

        for pred in TPFP_table:
            if pred[3] == 'easy':
                if pred[1] == 'TP':
                    TP_easy += 1
                else:
                    FP_easy += 1
                p_easy = TP_easy / (TP_easy + FP_easy)
                r_easy = TP_easy / easy_counts

                precision_easy.append(p_easy)
                recall_easy.append(r_easy)
            elif pred[3] == 'medium':
                if pred[1] == 'TP':
                    TP_medium += 1
                else:
                    FP_medium += 1
                p_medium = TP_medium / (TP_medium + FP_medium)
                r_medium = TP_medium / medium_counts

                precision_medium.append(p_medium)
                recall_medium.append(r_medium)
            elif pred[3] == 'hard':
                if pred[1] == 'TP':
                    TP_hard += 1
                else:
                    FP_hard += 1
                p_hard = TP_hard / (TP_hard + FP_hard)
                r_hard = TP_hard / hard_counts

                precision_hard.append(p_hard)
                recall_hard.append(r_hard)

            if pred[1] == 'TP':
                y_true.extend([pred[4]])
                y_pred.extend([pred[5]])

        return precision_easy, recall_easy, precision_medium, recall_medium, precision_hard, recall_hard, (y_true, y_pred)
