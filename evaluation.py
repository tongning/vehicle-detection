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
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


def plotPR(mode = '3D', threshold = 1.5, color = 'r'):
    p_easy, r_easy, p_medium, r_medium, p_hard, r_hard, _ = PR(mode, threshold)
    plt.plot(r_easy, p_easy, linestyle = '-', color=color)
    plt.plot(r_medium, p_medium, linestyle = '--', color=color)
    plt.plot(r_hard, p_hard, linestyle = ':', color=color)

def plotConfusion(mode = '3D', threshold = 1.5):
    _, _, _, _, _, _, orientations = PR(mode, threshold)
    show_conf_mat_heatmap(orientations, mode + " " + str(threshold))


def main(argv):


    axes = plt.gca()
    axes.set_xlim([0, 1.2])
    axes.set_ylim([0, 1.2])

    #plotPR('3D', 1.5, 'r')
    #plotPR('3D', 3, 'b')
    #plotPR('2D', 0.7, 'b')



    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.xlim(0,1.0)
    plt.ylim(0,1.05)
    plt.title('Precision-Recall curve for each IoU threshold (blue)\n and distance threshold (red)')
    plt.show()


    plt.figure()
    plotConfusion('3D', 1.5)

def show_conf_mat_heatmap(orientation_tuple, title):
    lbls = [0,1,2,3]
    conf_mat = confusion_matrix(orientation_tuple[0], orientation_tuple[1], labels = lbls)
    df = pd.DataFrame(conf_mat, lbls, lbls)
    sn.heatmap(df, annot=True, fmt='g')
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix orientations at " + title)
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
