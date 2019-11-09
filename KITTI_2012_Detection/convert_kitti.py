import os

in_dir = "/Users/ericchan/vehicle-detection/KITTI_2012_Detection/data_object_image_2/training/label_2"
out_dir = "/Users/ericchan/vehicle-detection/KITTI_2012_Detection/data_object_image_2/training_a/label_2"



labels = sorted(os.listdir(in_dir))

for label in labels:
    in_label = os.path.join(in_dir, label)
    out_label = os.path.join(out_dir, label)


    with open (in_label, 'r') as in_file:
        with open (out_label, 'w+') as out_file:
            for line in in_file.readlines():
                line_list = line.split(' ')
                if line_list[0] == "Car":
                    pi = 3.14159
                    alpha = int(round((float(line_list[3])+pi)*7/(2*pi)))
                    line_list[0] = "Car_" + str(alpha)

                    line_list[4] = str(int(round(float(line_list[4]))))
                    line_list[5] = str(int(round(float(line_list[5]))))
                    line_list[6] = str(int(round(float(line_list[6]))))
                    line_list[7] = str(int(round(float(line_list[7]))))

                    out_file.write(' '.join(line_list))
