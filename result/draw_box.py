import numpy as np
#import palettable as palettable
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator

plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }

fig = plt.figure(figsize=(20, 10))
#AP
iters = list(range(7))
alldata_AP = []  # 所有纵坐标数据
data_ANN = np.array([0.269,0.266,0.277,0.272,0.274])  # 单个数据
alldata_AP.append(data_ANN)
data_XBoost = np.array([0.253,0.246,0.248,0.248,0.245])
alldata_AP.append(data_XBoost)
data_PCBEP = np.array([0.421, 0.440, 0.450, 0.458, 0.443])
alldata_AP.append(data_PCBEP)
alldata_AP = np.array(alldata_AP)
allg = ["ANN", "XBoost", "PCBEP"]
# for i in range(2):
db = []
ax = fig.add_subplot(1, 3, 1)  # 画布
db = alldata_AP
df = pd.DataFrame(db.T, columns=allg)
df.boxplot(ax=ax, fontsize=26,showmeans=True,patch_artist=True,showfliers=False,showcaps=None,
           widths = 0.35, #箱体宽度
           boxprops={'color': 'black', 'facecolor': 'red'},#箱体边框颜色，内部填充色
           medianprops={'linestyle': '--', 'color': 'blue'}, # 设置中位数线的属性，线的类型和颜色
           meanprops={'marker':'D','markersize':10,'markerfacecolor':'black', 'markeredgecolor':'r','markeredgewidth':1.0})# 设置mark)

ax.set_title("AP", fontsize=24)
# ax.set_xlabel('Algorithms', fontsize=22)
ax.set_ylabel('value', fontsize=22)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
plt.ylim(0.24,0.46)

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
ax = fig.add_subplot(1, 3, 3)  # 画布
db = alldata_AUC
df = pd.DataFrame(db.T, columns=allg)
df.boxplot(ax=ax, fontsize=26,showmeans=True,patch_artist=True,showfliers=False,showcaps=None,
           widths = 0.35, #箱体宽度
           boxprops={'color': 'black', 'facecolor': 'red'},#箱体边框颜色，内部填充色
           medianprops={'linestyle': '--', 'color': 'blue'}, # 设置中位数线的属性，线的类型和颜色
           meanprops={'marker':'D','markersize':10,'markerfacecolor':'black', 'markeredgecolor':'r','markeredgewidth':1.0})# 设置mark)

ax.set_title("AUC", fontsize=24)
# ax.set_xlabel('Algorithms', fontsize=22)
ax.set_ylabel('value', fontsize=22)
plt.ylim(0.5,0.74)
plt.show()
