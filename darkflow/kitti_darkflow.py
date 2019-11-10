from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os




def boxing(original_img, predictions):
    newImage = np.copy(original_img)
    print(predictions)
    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)

    return newImage


options = {"model": "cfg/yolo-kitti.cfg",
           "load": -1,
           "threshold": 0.25,
           "gpu": 0.8}

tfnet = TFNet(options)


directory_l = "../VOC2012/JPEGImages/"

for i, filename in enumerate(sorted(os.listdir(directory_l))):
    if filename.endswith('.png'):
        filename_l = os.path.join(directory_l, filename)
        frame = cv2.imread(filename_l)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tfnet.return_predict(frame)
        _, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(boxing(frame, results))
        plt.show()
        #plt.pause(0.1)
        #plt.draw()

#%matplotlib inline
plt.show()
