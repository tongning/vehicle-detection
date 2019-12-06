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
    med, = plt.plot(r_medium, p_medium, color='g')
    hard, = plt.plot(r_hard, p_hard, color='b')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title("PRC for each KITTI difficulty @ {0}m, 3D".format(threshold))
    plt.xlim(0,1.0)
    plt.ylim(0,1.05)

    plt.legend((easy, med, hard), ("Easy", "Medium", "Hard"))

    plt.savefig("PRC_Difficulty.png", bbox_inches='tight')
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
    plt.title("PRC for different distance thresholds, 3D")
    plt.xlim(0,1.0)
    plt.ylim(0,1.05)

    plt.legend((plt1, plt2, plt3, plt4), ("10m", "5m", "3m", "1.5m"))

    plt.savefig("PRC_Distance.png", bbox_inches='tight')
    plt.show()

def plotOrientation(mode = '3D', threshold = 1.5):
    plt.clf()
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
    plt.title("PRC for 2D ({0} IoU) and 3D ({1}m distance)".format(iou, threshold))
    plt.xlim(0,1.0)
    plt.ylim(0,1.05)

    plt.legend((plt1, plt2), ("2D", "3D"))
    plt.savefig("PRC_2Dvs3D.png", bbox_inches='tight')
    #plt.show()

# Make a table that has the MAP values for each difficulty and for each distance threshold
def tableMAP():

    difficulties = ["Easy", "Medium", "Hard"]
    thresholds = [10, 5, 3, 1.5]
    df = pd.DataFrame()

    for threshold in thresholds:
        p_easy, r_easy, p_med, r_med, p_hard, r_hard, _ = PR('3D', threshold)

        map_easy = round(MAP(p_easy, r_easy), 4)
        map_med = round(MAP(p_med, r_med), 4)
        map_hard = round(MAP(p_hard, r_hard), 4)

        map_column = [map_easy, map_med, map_hard]

        df.insert(loc = 0, column = threshold, value = map_column)

    df.insert(loc = 0, column = "MAP", value = difficulties)
    df.set_index("MAP")

    # Make the table
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText = df.values, colLabels=df.columns, loc='center', fontsize=50)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    fig.tight_layout()

    plt.savefig("MAP.png")
    plt.show()


def plotSingle(mode = '3D', threshold = 1.5):

    plt.clf()

    p, r, _ = PR(mode, threshold, False)
    plt1, = plt.plot(r, p, color='r')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title("PRC at {0}".format(threshold))
    plt.xlim(0,1.0)
    plt.ylim(0,1.05)

    plt.savefig("PRC_Single.png")
    plt.show()


def main(argv):
    plotDifficulty()
    #plotDistance()
    plotOrientation()
    #plot2DComparison()
    #tableMAP()
    #plotSingle()


if __name__ == '__main__':
    main(sys.argv)
