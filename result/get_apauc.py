import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score
def ap_auc(file):
    y = []
    y_pred = []
    fr = open(file,'r')
    list = fr.readlines()
    for i in range(len(list)):
        newlist = list[i].split()
        y.append(float(newlist[0]))
        y_pred.append(float(newlist[1]))
    #y = np.array(y)
    #y_pred = np.array(y_pred,dtype=floacct)
    y = torch.tensor(y)
    y = y.squeeze()
    y_pred = torch.tensor(y_pred)
    y_pred = y_pred.squeeze()
    ap_score = metrics.average_precision_score(y,y_pred)
    auc_score = metrics.roc_auc_score(y,y_pred)
    return ap_score,auc_score

for i in range(0,5):
    ap, auc = ap_auc('724_ln2_result_(' + str(i) + ').txt')
    print(ap,auc)
for i in range(0,5):
    ap, auc = ap_auc('272_ln2_result_(' + str(i) + ').txt')
    print(ap,auc)

"""E:\Anaconda3\envs\bcell\python.exe E:\1Code\PCBEP\result\get_apauc.py 
0.2272928911736107 0.7109022989746356
0.21657507340284282 0.7092255657492355
0.19484537370630828 0.6966613707501349
0.16529019521839144 0.6730647886310488
0.1929975825887449 0.6985516171973376
0.2734304738745883 0.6706796020863965
0.26586705993186877 0.6869396349277889
0.18184668434263032 0.6380341409488474
0.18986364852896906 0.6240090196589472
0.20039953773778169 0.6324910769195263

Process finished with exit code 0
"""