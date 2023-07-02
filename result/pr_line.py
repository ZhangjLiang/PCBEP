import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
list_A = []
list_B = []
for i in range(0,5):
    fr = open('272_ln2_result_('+str(i)+').txt', 'r')
    list = fr.readlines()
    for i in range(len(list)):
        list_A.append(list[i].split()[0])
        list_B.append(list[i].split()[1])
plt.title('P-R Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
y_true = np.array(list_A,dtype=float)
y_scores = np.array(list_B,dtype=float)
# y_true为样本实际的类别1为正例0为反例，y_scores为阈值
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
plt.plot(recall, precision)
plt.savefig('./picture/272prline.png')
plt.show()
