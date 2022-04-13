import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score, roc_curve, roc_auc_score, auc

"""
For each model there has to be a txt file with the gt values gt_modelname.txt and one txt file with the hm_max_values hm_modelname.txt.
(gt values and hm_max_values are computed in the method test() within the gaze_prediction_and_evaluation.py file)
"""

models_filename = ['yolo5', 'yolo3', 'centertrack', 'bdda', 'dreyeve', 'mlnet', 'picanet', 'baseline']
models_plot = ['YOLOv5', 'YOLOv3', 'CenterTrack', 'BDD-A', 'DR(eye)VE', 'ML-Net', 'PiCANet', 'Baseline']

dataset = ["BDDA", "DREYEVE"]

for i in range(len(models_filename)):
    mod = models_filename[i]
    nm = models_plot[i]
    gt = []
    hm = []
    try:
        with open("gt_"+mod+".txt", "r") as f:
            for line in f:
                gt.append(float(line.strip()))

        with open("hm_"+mod+".txt", "r") as f:
            for line in f:
                hm.append(float(line.strip()))
    except:
        continue

    print('AUC:'+nm, roc_auc_score(gt, hm))
    fpr, tpr, threshold = roc_curve(gt, hm)


    gmeans = np.sqrt(tpr * (1-fpr))
    optimal_idx = np.argmax(gmeans)
    print('Best Threshold=%f' % threshold[optimal_idx])

    plt.plot(fpr, tpr, label = nm + ", Th = %.2f" % threshold[optimal_idx])
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], marker='X', markersize=7, color="black")


plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0.0,1.0],[1.0,0.0],color='black',linestyle='dashed', linewidth=0.5)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('./roc.png')
