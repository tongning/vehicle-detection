import subprocess
import sys
import os
import numpy as np

# Class for encapsulating a single identified obstacle in an image.
class Obstacle:

    def __init__(self, classification, prob, image_no, top, bot, left, right):
        # What is it? A bicycle, a banana, a car?
        self.classification = classification
        # What is yolo's probability that it thinks it's this class?
        self.prob = prob
        # Which cropped image was it when it was extracted from the full image?
        self.image_no = image_no
        # Dimensions of the bounding box around the obstacle.
        self.top = int(top)
        self.bot = int(bot)
        self.left = int(left)
        self.right = int(right)

    def get_classification(self):
        return self.classification
    def get_prob(self):
        return self.prob
    def get_image_no(self):
        return self.image_no
    def get_dims(self):
        return (self.top, self.bot, self.left, self.right)

# Process a single image and return a list of obstacle objects.
def process_image(image_path):
    base_path = "/home/anthony/darknet/"

    # Have to change dirs because yolo(darknet) has to be run from within it's directory.
    # Looks to be a known issue people are complaining about.
    os.chdir(base_path)

    # TODO: Suppress whatever output is 1,2,3 ... 106 yolo.
    # TODO: When it's loading weights, can I hold that in memory for multiple image processes?
    program_path = base_path + "darknet"
    args_list = ["detect", base_path + "cfg/yolov3.cfg", base_path + "yolov3.weights", image_path]

    # Get the output from our program and decode it into a UTF-8 string
    output = subprocess.check_output([program_path] + args_list).decode('utf-8')

    obstacles = []

    # Looking at lines 1-end to get our obstacles.
    lines = output.split("\n")[1:]
    # Filter out empty strings (generally at end)
    lines = list(filter(None, lines))
    num_lines = len(lines)

    # Create all the obstacle objects.
    for i in range(0,num_lines,2):
        first_line = lines[i]
        second_line = lines[i+1]
        first_split = first_line.split(": ")
        second_split = second_line.split(",")
        print(first_split)
        print(second_split)

        classification = first_split[0]
        prob = float(first_split[1].strip("%")) * 0.01
        image_no = second_split[1]
        top = second_split[3]
        bot = second_split[5]
        left = second_split[7]
        right = second_split[9]

        obstacle = Obstacle(classification, prob, image_no, top, bot, left, right)
        obstacles.append(obstacle)

    return obstacles

def get_obstacle_dims(left_image_path):

    obstacles = process_image(left_image_path)

    dims_list = []

    for obstacle in obstacles:
        dims_list.append(obstacle.get_dims())

    return dims_list

def get_obstacle_info(left_image_path):

    obstacles = process_image(left_image_path)

    info_list = []

    # Classifications that we care about
    classifications = ["car", "truck", "bicycle", "person", "motorbike", "bus"]

    for obstacle in obstacles:
        #if obstacle.classification in classifications:
        six_tuple = (obstacle.top, obstacle.bot, obstacle.left, obstacle.right, obstacle.classification, obstacle.prob)
        info_list.append(six_tuple)

    return info_list

def main():

    #images_path = sys.argv[1]

    # TODO: BUG at 0000-left/000016.png. For some reason outputs a car without dimensions print...
    # Creating a pickled representation of all the bounding boxes from a stream of ~300 images.
    for i in range(11, 15):
        images_path = "/home/anthony/git/vehicle-detection/kitti/00{:02d}-left/".format(i)
        #images_path = "/home/anthony/git/vehicle-detection/kitti/0010-left/"

        info_list = []
        left_files = sorted(os.listdir(images_path))

        for file in left_files:
            left_image_file_path = images_path + file
            obstacles = get_obstacle_info(left_image_file_path)
            info_list.append(obstacles)
            print(left_image_file_path)

        nparray = np.array([np.array(obstacles) for obstacles in info_list])
        save_path = "/home/anthony/git/vehicle-detection/" + images_path.split("/")[-2] + ".pyc"
        np.save(save_path, nparray, allow_pickle=True)

if __name__ == "__main__":
     main()



# OLD MAIN (For printing obstacles from a single image)
# def main():

    # image_path = sys.argv[1]
    # obstacles = process_image(image_path)
    # dims_list = []
    #
    # for obstacle in obstacles:
    #     dims_list.append(obstacle.get_dims())
    #
    # return dims_list

    #for obstacle in obstacles:
    #    print ('Image:{0} Dims:{1}'.format(obstacle.get_image_no(), obstacle.get_dims()))



