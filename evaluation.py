###############################################################################
#
# Before you run this, use generate.py to create some data files. After
# generating files, you should have some sequences in vehicle-detection/eval.
# When you run this evaluation file, it will evaluate on all of the sequences
# in vehicle-detection/eval. Adjust the thresholds and plots below as needed.
#
###############################################################################


import sys
sys.path.append('./evaluation')
from precisionrecall import *
import matplotlib.pyplot as plt

def main(argv):
    # 2D with an iou threshold of 50%, 60%, 70%, 80%
    precision05, recall05 = PR('2D', 0.5)
    precision06, recall06 = PR('2D', 0.6)
    precision07, recall07 = PR('2D', 0.7)
    precision08, recall08 = PR('2D', 0.8)

    # 3D with a distance threshold of 10m, 7m, 5m, 3m
    precision310, recall310 = PR('3D', 10)
    precision307, recall307 = PR('3D', 7)
    precision305, recall305 = PR('3D', 5)
    precision303, recall303 = PR('3D', 3)
    precision302, recall302 = PR('3D', 2)
    precision301, recall301 = PR('3D', 1)


    axes = plt.gca()
    axes.set_xlim([0, 1.2])
    axes.set_ylim([0, 1.2])
    plt.plot(recall05, precision05, 'b')
    plt.plot(recall06, precision06, 'b')
    plt.plot(recall07, precision07, 'b')
    plt.plot(recall08, precision08, 'b')

    plt.plot(recall310, precision310, 'r')
    plt.plot(recall307, precision307, 'r')
    plt.plot(recall305, precision305, 'r')
    plt.plot(recall303, precision303, 'r')
    plt.plot(recall302, precision302, 'r')
    plt.plot(recall301, precision301, 'r')
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
