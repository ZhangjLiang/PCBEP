from matplotlib import pyplot as plt
import numpy as np
import numpy as np
#import palettable as palettable
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator

color=['k', 'g', 'r', 'deepskyblue']

alldata_AP = []  # 所有纵坐标数据
data_ANN = np.array([0.269,0.266,0.277,0.272,0.274])  # 单个数据
alldata_AP.append(data_ANN)
data_XBoost = np.array([0.253,0.246,0.248,0.248,0.245])
alldata_AP.append(data_XBoost)
data_PCBEP = np.array([0.421, 0.440, 0.450, 0.458, 0.443])
alldata_AP.append(data_PCBEP)
alldata_AP = np.array(alldata_AP)
allg = ["ANN", "XBoost", "PCBEP"]

db = []
db = alldata_AP
df = pd.DataFrame(db.T, columns=allg)

plt.figure(figsize=(8, 8))
c_list = ['#ef476f', '#ffd166', '#118AD5', '#20B2AA', '#FFA500', '#9370DB','#98FB98','#1E90FF','#7CFC00',
          '#FFFF00']  # 颜色代码列表
# 绘制箱线图
f = plt.boxplot(df,vert=True, sym='+b',  showmeans=True,
                meanline=True, patch_artist=True, widths=0.5)
for box, c in zip(f['boxes'], c_list):  # 对箱线图设置颜色
    box.set(color=c, linewidth=2)
    box.set(facecolor=c)
for i in range(len(db)):
    y = db[i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.8,color = 'black')
plt.title("AP", fontsize=24)
plt.ylabel('value', fontsize=22)
plt.ylim(0.24,0.46)
plt.savefig('ap_box.png')
plt.show()

plt.figure(figsize=(8, 8))
#AUC
alldata_AUC = []  # 所有纵坐标数据
data_ANN = np.array([0.573,0.569,0.576,0.575,0.567])  # 单个数据
alldata_AUC.append(data_ANN)
data_XBoost = np.array([0.580,0.573,0.579,0.578,0.567])
alldata_AUC.append(data_XBoost)
data_PCBEP = np.array([0.696,0.724,0.733,0.716,0.739])
alldata_AUC.append(data_PCBEP)
alldata_AUC = np.array(alldata_AUC)
allg = ["ANN", "XBoost", "PCBEP"]
# for i in range(2):
db = []
db = alldata_AUC
df = pd.DataFrame(db.T, columns=allg)
f2 = plt.boxplot(df,vert=True, sym='+b',  showmeans=True,
                meanline=True, patch_artist=True, widths=0.5)
for box, c in zip(f2['boxes'], c_list):  # 对箱线图设置颜色
    box.set(color=c, linewidth=2)
    box.set(facecolor=c)
for i in range(len(db)):
    y = db[i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.8,color = 'black')
plt.title("AUC", fontsize=24)
plt.ylabel('value', fontsize=22)
plt.ylim(0.5,0.74)
plt.savefig('auc_box.png')
plt.show()



