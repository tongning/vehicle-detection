import sys
#sys.path.append("..")

from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os




def boxing(original_img, predictions):
    newImage = np.copy(original_img)
    #print(predictions)
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



def main(argv):
    options = {"model": "/home/eric/vehicle-detection/network/cfg/kitti.cfg",
               #"model": "cfg/yolo.cfg",
               "load": -1,
               #"load": "/home/eric/vehicle-detection/network/weights/yolo.weights",
               "threshold": 0.35,
               "gpu": 0.8}

    #os.chdir("darkflow")
    tfnet = TFNet(options)

    plt.ion()
    plt.show()

    if len(argv) > 1:
        sequence_number = argv[1]
    else:
        sequence_number = "0000"

    directory_l = os.path.join("/home/eric/vehicle-detection/data/KITTI-Tracking/testing/image_02/", sequence_number)
    _, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(cv2.imread(os.path.join(directory_l, '000000.png')))
    for i, filename in enumerate(sorted(os.listdir(directory_l))):
        if filename.endswith('.png'):
            filename_l = os.path.join(directory_l, filename)
            frame = cv2.imread(filename_l)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tfnet.return_predict(frame)
            #ax.imshow(boxing(frame, results))
            im.set_data(boxing(frame, results))
            #plt.show()
            plt.pause(0.01)
            plt.draw()

    #%matplotlib inline
    plt.show()




if __name__ == "__main__":
    main(sys.argv)
