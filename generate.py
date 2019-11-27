################################################################################
#
# Use this file to generate prediction and groundtruth files for sequences.
# Usage:
# python generate.py 0010 visualize
# python generate.py visualize 0011 0012 0017
# python generate.py all
#
################################################################################




import sys
sys.path.append('./evaluation')
import os
from neuralnetprediction import *
from groundtruth import *
from visualize2d import PlaySequence

def main(argv):
    if 'all' in argv:
        sequences = [str(seq_num).zfill(4) for seq_num in range(0, 21)]
    else:
        sequences = [arg for arg in argv[1:] if arg != 'visualize' and arg != 'all']

    if 'visualize' in argv:
        visualize = True
    else:
        visualize = False


    vd_path = os.getcwd()
    gtparser = GroundTruthParser()
    model = NetworkModel()

    print("Running Model")

    for sequence_name in sequences:
        gtparser.OutGroundTruthSequence(sequence_name)
        model.PredictSequence(sequence_name, visualize)

        #if visualize:
        #    PlaySequence(sequence_name, vd_path)

    os.chdir(vd_path)

if __name__ == '__main__':
    main(sys.argv)
