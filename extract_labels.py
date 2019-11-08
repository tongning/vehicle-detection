# Hard code everything for now.
import numpy as np

def main():

    file_path = "/home/anthony/git/vehicle-detection/kitti/labels/training/label_02/0010.txt"
    list_of_dims_3D = []

    old_image_no = 0
    image_dims_list = []
    with open(file_path, 'r') as file:

        line = file.readline()
        while line:
            split_line = line.split(" ")
            image_no = int(split_line[0])

            if image_no != old_image_no:
                list_of_dims_3D.append(image_dims_list)
                image_dims_list = []

            if split_line[2] == "Car":
                alpha = float(split_line[5])
                left = float(split_line[6])
                top = float(split_line[7])
                right = float(split_line[8])
                bot = float(split_line[9])
                height_3D = float(split_line[10])
                width_3D = float(split_line[11])
                length_3D = float(split_line[12])
                location_x = float(split_line[13])
                location_y = float(split_line[14])
                location_z = float(split_line[15])
                rotation_y = float(split_line[16].strip("\n"))
                # top, bot, left, right to match dims from get_dims.py
                dims_3D = [alpha, top, bot, left, right, height_3D, width_3D, length_3D, location_x, location_y, location_z, rotation_y]
                image_dims_list.append(dims_3D)

            old_image_no = image_no
            line = file.readline()

    nparray = np.array(list_of_dims_3D)

    save_path = "/home/anthony/git/vehicle-detection/dims_labels"

    np.save(save_path, nparray, allow_pickle=True)

if __name__ == "__main__":
    main()