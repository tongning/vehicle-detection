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



def ReadGroundTruthSequence(in_sequence_filename):
    # get a list of ground truth labels for a sequence
    sequence = [[] for frame in range(1200)]

    with open (in_sequence_filename, 'r') as in_sequence_file:
        for line in in_sequence_file.readlines():
            line = line.split(' ')
            if line[2] == 'Car' or line[2] == 'Van':
                tracked_object = {}
                tracked_object['frame_number'] = int(line[0])
                tracked_object['type'] = 'Car_' + _DiscretizeAlpha(float(line[5]))
                tracked_object['bbox'] = {'left': int(float(line[6])), 'top': int(float(line[7])), 'right': int(float(line[8])), 'bottom': int(float(line[9]))}
                #[float(i) for i in line[6:10]]
                tracked_object['3dbbox_dim'] = [float(i) for i in line[10:13]]
                tracked_object['3dbbox_loc'] = [float(i) for i in line[13:16]]
                tracked_object['rotation_y'] = float(line[16])
                tracked_object['alpha'] = float(line[5])
                tracked_object['occluded'] = int(line[4])

                #if tracked_object['frame_number'] >= len(sequence):
                #    sequence.append([tracked_object])
                #else:
                sequence[tracked_object['frame_number']].append(tracked_object)
    return sequence

def OutGroundTruthSequence(sequence_name, out_directory = '/home/eric/vehicle-detection/evaluation'):
    # Read objects ground truth sequence file. Write one file per frame.
    in_sequence_filename = os.path.join("/home/eric/vehicle-detection/data/KITTI-tracking/training/label_02/", sequence_name + '.txt')
    out_directory=os.path.join(out_directory, 'eval', sequence_name, 'groundtruth')
    sequence = ReadGroundTruthSequence(in_sequence_filename)
    os.makedirs(out_directory, exist_ok=True)
    for frame_number, frame in enumerate(sequence):
        out_file_name = os.path.join(out_directory, str(frame_number).zfill(6))
        with open(out_file_name, 'wb+') as out_file_frame:
            pickle.dump(frame, out_file_frame, pickle.HIGHEST_PROTOCOL)

#OutGroundTruthSequence('/home/eric/vehicle-detection/data/KITTI-tracking/training/label_02/0010.txt', '0010/groundtruth')

# EXAMPLE
#seq = ReadGroundTruthSequence('/home/eric/vehicle-detection/data/KITTI-tracking/training/label_02/0010.txt')
#print(seq[0])
