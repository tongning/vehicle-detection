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

    # # 2D with an iou threshold of 50%, 60%, 70%, 80%
    # precision05, recall05 = PR('2D', 0.5)
    # precision06, recall06 = PR('2D', 0.6)
    # precision07, recall07 = PR('2D', 0.7)
    # precision08, recall08 = PR('2D', 0.8)
    #
    # # 3D with a distance threshold of 10m, 7m, 5m, 3m
    # precision310, recall310, orientation310 = PR('3D', 10)
    # precision307, recall307, orientation307 = PR('3D', 7)
    # precision305, recall305, orientation305 = PR('3D', 5)
    # precision303, recall303, orientation303 = PR('3D', 3)
    # precision302, recall302, orientation302 = PR('3D', 2)
    # precision301, recall301, orientation301 = PR('3D', 1)
    #
    # # Show heatmaps
    # show_conf_mat_heatmap(orientation310, "distance 10m")
    # show_conf_mat_heatmap(orientation307, "distance 7m")
    # show_conf_mat_heatmap(orientation305, "distance 5m")
    # show_conf_mat_heatmap(orientation303, "distance 3m")
    # show_conf_mat_heatmap(orientation302, "distance 2m")
    # show_conf_mat_heatmap(orientation301, "distance 1m")
    #
    # # Show AUPRC curves of different IOU thresholds.
    # plt.plot(recall05, precision05, 'b')
    # plt.plot(recall06, precision06, 'b')
    # plt.plot(recall07, precision07, 'b')
    # plt.plot(recall08, precision08, 'b')
    #
    # # Show AUPRC curves of different distance thresholds.
    # plt.plot(recall310, precision310, 'r')
    # plt.plot(recall307, precision307, 'r')
    # plt.plot(recall305, precision305, 'r')
    # plt.plot(recall303, precision303, 'r')
    # plt.plot(recall302, precision302, 'r')
    # plt.plot(recall301, precision301, 'r')


    #p_easy, r_easy, p_medium, r_medium, p_hard, r_hard = PR('3D', 1.5)
    #plt.plot(r_easy, p_easy, 'r')
    #plt.plot(r_medium, p_medium, 'g')
    #plt.plot(r_hard, p_hard, 'b')
    #plotPR('3D', 1.5, 'r')
    #plotPR('2D', 0.7, 'b')
    plotConfusion('2D', 0.7)

    axes = plt.gca()
    axes.set_xlim([0, 1.2])
    axes.set_ylim([0, 1.2])

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.xlim(0,1.0)
    plt.ylim(0,1.05)
    plt.title('Precision-Recall curve for each IoU threshold (blue)\n and distance threshold (red)')
    plt.show()

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
