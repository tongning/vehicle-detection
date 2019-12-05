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

def plotDifficulty(mode = '3D', threshold = 1.5):
    p_easy, r_easy, p_medium, r_medium, p_hard, r_hard, _ = PR(mode, threshold)
    easy, = plt.plot(r_easy, p_easy, color='r')
    #print("MAP: {0}".format(MAP(p_easy, r_easy)))
    med, = plt.plot(r_medium, p_medium, color='g')
    #print("MAP: {0}".format(MAP(p_medium, r_medium)))
    hard, = plt.plot(r_hard, p_hard, color='b')
    #print("MAP: {0}".format(MAP(p_hard, r_hard)))

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title("AUPRC for each KITTI difficulty @ {0}m, 3D".format(threshold))
    plt.xlim(0,1.0)
    plt.ylim(0,1.05)

    plt.legend((easy, med, hard), ("Easy", "Medium", "Hard"))

    plt.savefig("AUPRC_Difficulty.png", bbox_inches='tight')
    plt.show()

def plotDistance(mode = '3D'):
    p10, r10, _ = PR(mode, 10, False)
    p5, r5, _ = PR(mode, 5, False)
    p3, r3, _ = PR(mode, 3, False)
    p15, r15, _ = PR(mode, 1.5, False)

    plt1, = plt.plot(r10, p10, color='r')
    plt2, = plt.plot(r5, p5, color = 'g')
    plt3, = plt.plot(r3, p3, color = 'b')
    plt4, = plt.plot(r15, p15, color = 'purple')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title("AUPRC for different distance thresholds, 3D")
    plt.xlim(0,1.0)
    plt.ylim(0,1.05)

    plt.legend((plt1, plt2, plt3, plt4), ("10m", "5m", "3m", "1.5m"))

    plt.savefig("AUPRC_Distance.png", bbox_inches='tight')
    plt.show()

def plotOrientation(mode = '3D', threshold = 1.5):
    _, _, orientation = PR(mode, threshold, False)
    word_lbls = ["front","l-diag","side","r-diag"]
    lbls = [0,1,2,3]
    conf_mat = confusion_matrix(orientation[0], orientation[1], labels = lbls)
    df = pd.DataFrame(conf_mat, word_lbls, word_lbls)
    sn.heatmap(df, annot=True, fmt='g')
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix orientations at {0}m, 3D".format(threshold))

    plt.savefig("Orientation_ConfusionMatrix.png", bbox_inches='tight')
    plt.show()

def plot2DComparison(threshold = 1.5):
    iou = 0.7
    p_2D, r_2D, _ = PR('2D', iou, False)
    p_3D, r_3D, _ = PR('3D', threshold, False)

    plt1, = plt.plot(r_2D, p_2D, color = 'r')
    plt2, = plt.plot(r_3D, p_3D, color = 'b')


    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title("AUPRC for 2D ({0} IoU) and 3D ({1}m distance)".format(iou, threshold))
    plt.xlim(0,1.0)
    plt.ylim(0,1.05)

    plt.legend((plt1, plt2), ("2D", "3D"))
    plt.savefig("AUPRC_2Dvs3D.png", bbox_inches='tight')
    plt.show()


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
    #plotConfusion('2D', 0.7)


    #plotPR('3D', 1.5, 'r')
    #plotPR('3D', 3, 'b')
    #plotPR('2D', 0.7, 'b')



    # plt.xlabel('Recall', fontsize=14)
    # plt.ylabel('Precision', fontsize=14)
    # plt.xlim(0,1.0)
    # plt.ylim(0,1.05)
    # plt.title('Precision-Recall curve for each IoU threshold (blue)\n and distance threshold (red)')
    # plt.show()
    #
    #
    # plt.figure()
    #plotConfusion('3D', 1.5)

    plotDifficulty()
    plotDistance()
    plotOrientation()
    plot2DComparison()


if __name__ == '__main__':
    main(sys.argv)
