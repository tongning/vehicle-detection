import os
import pickle

def _DiscretizeAlpha(alpha):
    pi = 3.14159
    range = 2*pi/16

    if abs(alpha - pi/2) <= range or abs(alpha + pi/2) <= range:
        angle = 0
    elif abs(alpha - 3*pi/4) <= range or abs(alpha + pi/4) <= range:
        angle = 1
    elif abs(alpha - 0) <= range or abs(alpha - pi) <= range or abs(alpha + pi) <= range:
        angle = 2
    elif abs(alpha - pi/4) <= range or abs(alpha + 3*pi/4) <= range:
        angle = 3
    return str(angle)

class GroundTruthParser:
    def __init__(self):
        self.vd_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


    def ReadGroundTruthSequence(self, sequence_number):
        in_sequence_filepath = os.path.join(self.vd_directory, 'data/KITTI-tracking/training/label_02', sequence_number + '.txt')

        # get a list of ground truth labels for a sequence
        sequence = [{'tracked_objects':[]} for frame in range(1200)]

        with open (in_sequence_filepath, 'r') as in_sequence_file:
            for line in in_sequence_file.readlines():
                line = line.split(' ')
                if line[2] == 'Car' or line[2] == 'Van':
                    tracked_object = {}
                    tracked_object['frame_number'] = int(line[0])
                    tracked_object['type'] = 'Car_' + _DiscretizeAlpha(float(line[5]))
                    tracked_object['bbox'] = {'left': int(float(line[6])), 'top': int(float(line[7])), 'right': int(float(line[8])), 'bottom': int(float(line[9]))}
                    tracked_object['3dbbox_dim'] = [float(i) for i in line[10:13]]
                    tracked_object['3dbbox_loc'] = [float(i) for i in line[13:16]]
                    tracked_object['rotation_y'] = float(line[16])
                    tracked_object['alpha'] = float(line[5])
                    tracked_object['occluded'] = int(line[4])
                    tracked_object['truncated'] = int(line[3])

                    if tracked_object['bbox']['bottom'] - tracked_object['bbox']['top'] >= 40 and tracked_object['occluded'] == 0 and tracked_object['truncated'] == 0:
                        tracked_object['difficulty'] = 'easy'
                    elif tracked_object['bbox']['bottom'] - tracked_object['bbox']['top'] >= 25 and tracked_object['occluded'] <= 1 and tracked_object['truncated'] <= 1:
                        tracked_object['difficulty'] = 'medium'
                    elif tracked_object['bbox']['bottom'] - tracked_object['bbox']['top'] >= 25 and tracked_object['occluded'] <= 2 and tracked_object['truncated'] <= 2:
                        tracked_object['difficulty'] = 'hard'
                    else:
                        tracked_object['difficulty'] = 'very hard'

Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %

                    sequence[tracked_object['frame_number']]['tracked_objects'].append(tracked_object)
        return sequence

    def OutGroundTruthSequence(self, sequence_number, out_directory=None):
        # Read objects ground truth sequence file. Write one file per frame.
        if not out_directory:
            out_directory = os.path.join(self.vd_directory, 'eval', sequence_number, 'groundtruth')
        sequence = self.ReadGroundTruthSequence(sequence_number)
        os.makedirs(out_directory, exist_ok=True)
        for frame_number, frame in enumerate(sequence):
            out_file_name = os.path.join(out_directory, str(frame_number).zfill(6))
            with open(out_file_name, 'wb+') as out_file_frame:
                pickle.dump(frame, out_file_frame, pickle.HIGHEST_PROTOCOL)

#OutGroundTruthSequence('/home/eric/vehicle-detection/data/KITTI-tracking/training/label_02/0010.txt', '0010/groundtruth')

# EXAMPLE
#seq = ReadGroundTruthSequence('/home/eric/vehicle-detection/data/KITTI-tracking/training/label_02/0010.txt')
#print(seq[0])
