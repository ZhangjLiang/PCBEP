import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator


# esm baseline
'''conv1_ap = ['0.3636790936177444','0.37076797208528955','0.3982374039721905','0.32896160288982057','0.30919042869688845']
conv1_auc = ['0.6052097781569337','0.6478183306748907','0.6733747020575841','0.6550302411893696','0.6546455855772056']
conv2_ap =  ['0.3641188142329076','0.3876879397889665','0.3571514199187948','0.38285035907749354','0.3697363584808345']
conv2_auc = ['0.6754236540240994','0.6833357417978294','0.6664411295145747','0.6804685760203921','0.6649870298842815']'''

# esm + baseline 59d,baseline
'''conv1_ap = ['0.3690603593409555','0.36991734538779936','0.3650578160460567','0.3211916180884673','0.34474209972763625']
conv1_auc = ['0.6676702973084256','0.6730221420106348','0.6657237543729818','0.6292988886176383','0.6316448879976152']
conv2_ap =  ['0.39550957491245725','0.31465210943091604','0.3326117701221128','0.3582252141299065','0.3548607728890652']
conv2_auc = ['0.6837278030145979','0.6266789233746554','0.6474556739629063','0.6577076076506667','0.6638354255082961']'''

# esm + baseline 1280d,baseline
conv1_ap = ['0.4286843067645899','0.4670209607560738','0.40512459881221','0.3936390498898113','0.44411802285934837']
conv1_auc = ['0.702104760389604','0.7206804361382221','0.6812351082461999','0.6924684196097426','0.7200646451618402']
conv2_ap =  ['0.4529552825109927','0.4400765434151439','0.4432425604865645','0.4226659465542555','0.42983466458785274']
conv2_auc = ['0.6877808854299096','0.6807556779361277','0.6966828835918334','0.6756463202257963','0.685505911246781']

# esm 1280 only
conv1_ap = ['0.42847123634295464','0.4242806403214661','0.396765657085834','0.4072735041811788','0.43850259733467567']
conv1_auc = ['0.6914165756388166','0.6819497171152719','0.677504321751371','0.6837533753694973','0.7044968693048408']
conv2_ap =  ['0.39368888727701956','0.4073331946476053','0.44664713248676635','0.45971926164782884','0.4244053022069938']
conv2_auc = ['0.6428496747855459','0.6655207870203992','0.7043201284948658','0.7047043417798655','0.6661773594205658']

# 724dataset esm 1280 + 59d
conv1_ap = ['0.43009066767699566','0.45288337305184134','0.47786987657684044','0.49159907115960033','0.45496400762150224']
conv1_auc = ['0.6963670256789302','0.7191899103283231','0.7374914956431617','0.7452726522483347','0.7203249387330725']
conv2_ap =  ['0.4753335275257431','0.47392973390449367','0.5383377510436806','0.4701400144614282','0.45778206934867555']
conv2_auc = ['0.7056980652657334','0.7168113256777132','0.7578071852708412','0.7170804110348268','0.6849074473117066']



def get_data(list):
    max_value = max(list)
    max_idx = list.index(max_value)
    min_value = min(list)
    min_idx = list.index(min_value)
    sum = 0
    for i in range(len(list)):
        sum+=float(list[i])
    ave = sum/5
    fudong = float(max_value) - float(ave)
    return ave,fudong

#todo need modify
x_data1 = ['Atom level+Residue moudle','baseline']
x_data2 = ['Residue level','baseline']
#todo need modify

#conv1
ave1,yerr1 = get_data(conv1_ap)
ave1 = round(float(ave1), 3)
yerr1 = round(float(yerr1), 3)
ave2,yerr2 = get_data(conv1_auc)
ave2 = round(float(ave2), 3)
yerr2 = round(float(yerr2), 3)

#conv2
ave3,yerr3 = get_data(conv2_ap)
ave3 = round(float(ave3), 3)
yerr3 = round(float(yerr3), 3)
ave4,yerr4 = get_data(conv2_auc)
ave4 = round(float(ave4), 3)
yerr4 = round(float(yerr4), 3)

#todo need modify
#conv1
y_data1,yerr1 = [ave1,0.403],[yerr1,0.015]
y_data2,yerr2 = [ave2,0.689],[yerr2,0.018]
#todo need modify
#conv2
y_data3,yerr3 = [ave3,0.403],[yerr3,0.015]
y_data4,yerr4 = [ave4,0.689],[yerr4,0.018]

# bcell

#todo need modify
#conv1
mask1 = [''+str(ave1)+'±'+str(yerr1[0])+'','0.403±0.015']
mask2 = [''+str(ave2)+'±'+str(yerr2[0])+'','0.689±0.018']
#conv2
mask3 = [''+str(ave3)+'±'+str(yerr3[0])+'','0.403±0.015']
mask4 = [''+str(ave4)+'±'+str(yerr4[0])+'','0.689±0.018']

# 900*600
plt.figure(figsize=(8, 6))
plt.rcParams["font.sans-serif"]=['SimHei']
plt.rcParams["axes.unicode_minus"]=False
color = {0: "#20B2AA", 1: "#FFA500", 2: "#9370DB", 3: "#98FB98", 4: "#1E90FF", 5: "#7CFC00", 6: "#FFFF00",
             7: "#808000", 8: "#FF00FF", 9: "#FA8072", 10: "#7B68EE", 11: "#9400D3", 12: "#800080", 13: "#A0522D",
             14: "#D2B48C", 15: "#D2691E", 16: "#87CEEB", 17: "#40E0D0", 18: "#5F9EA0",
             19: "#FF1493", 20: "#FF6347"}
for i in range(len(x_data1)):
    mpl.rcParams['font.size'] = 16
    plt.bar(x_data2[i], float(y_data4[i]), yerr=yerr4[i], width=0.2, align='center', color=color[i])
    plt.text(x_data2[i], float(y_data4[i]), mask4[i], fontsize=16)
#plt.ylabel("AP")
plt.ylabel("AUC")
plt.savefig('esm_724_4.png')
#plt.savefig('new_auc2_1280.png')

#for i in range(len(x_data2)):
#plt.bar(x_data[i],float(ap[i]),yerr = yerr[i],width=0.3,align='center')
#plt.text(x_data[i],float(ap[i]), mask[i], fontsize=6, color='red')

