



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
