from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
import numpy as np





def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)

    return newImage


options = {"model": "cfg/yolo.cfg",
           "load": "weights/yolo.weights",
           "threshold": 0.1,
           "gpu": 1.0}

tfnet = TFNet(options)

"""
directory_l      = "../data_tracking_image_2/testing/image_02/0020"
directory_r      = "../data_tracking_image_2/testing/image_03/0020"

for i, filename in enumerate(sorted(os.listdir(directory_l))):
    #return_value, frame = vid.read()
    #if return_value:
    #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #    image = Image.fromarray(frame)
    if filename.endswith('.png'):
        filename_l = os.path.join(directory_l, filename)
        filename_r = os.path.join(directory_r, filename)
        frame = cv2.imread(filename_l)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
"""

img = cv2.imread("../data_tracking_image_2/testing/image_02/0020/000000.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = tfnet.return_predict(img)

#%matplotlib inline

_, ax = plt.subplots(figsize=(20, 10))
ax.imshow(boxing(img, results))
plt.show()
