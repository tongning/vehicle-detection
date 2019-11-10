import os

in_dir =  "/Users/ericchan/vehicle-detection/KITTI_2012_Detection/data_object_image_2/training/label_2"
out_dir = "/Users/ericchan/vehicle-detection/KITTI_2012_Detection/data_object_image_2/training_a/label_2"
os.makedirs(out_dir, exist_ok = True)

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
                    #print(line_list)
                    #alpha = int(round((float(line_list[3])+pi)*7/(2*pi)))
                    line_list[0] = "Car_" + str(line_list[3])
                    out_file.write(' '.join(line_list))
                    print(line_list)
