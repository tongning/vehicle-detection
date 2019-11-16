import os
import sys

def convert_kitti(in_dir):
    labels = sorted(os.listdir(in_dir))
    for label in labels:
        in_label = os.path.join(in_dir, label)


        with open (in_label, 'r') as in_file:
            out_lines = []

            for line in in_file.readlines():
                line_list = line.split(' ')

                #if line_list[0] == "Pedestrian" or line_list[0] == "Cyclist":
                #    out_lines.append(' '.join(line_list))
                if line_list[0] == "Car" or line_list[0] == "Van":
                    pi = 3.14159
                    range = 2*pi/16
                    alpha = float(line_list[3])

                    if abs(alpha - pi/2) <= range or abs(alpha + pi/2) <= range:
                        angle = 0
                    elif abs(alpha - 3*pi/4) <= range or abs(alpha + pi/4) <= range:
                        angle = 1
                    elif abs(alpha - 0) <= range or abs(alpha - pi) <= range or abs(alpha + pi) <= range:
                        angle = 2
                    elif abs(alpha - pi/4) <= range or abs(alpha + 3*pi/4) <= range:
                        angle = 3

                    if line_list[0] == "Van":
                        line_list[0] = "Car"
                    line_list[0] += "_" + str(angle)
                    out_lines.append(' '.join(line_list))

        with open(in_label, 'w') as out_file:
            for line in out_lines:
                out_file.write(line)

def main(argv):
    convert_kitti(argv[0])


if __name__ == '__main__':
    main(sys.argv[1:])
