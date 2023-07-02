import os
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import matplotlib

matplotlib.use('TkAgg')


def get_data(path):
    data = []
    dic = {}
    for name in path:
        with open(name, 'r') as da:
            lines = da.readlines()
            # label = [float(line.strip().split()[0]) for line in lines]
            # score = [float(line.strip().split()[1]) for line in lines]
            if name.split('/')[-1].split('_')[1] == 'DLBEpitope':
                    #\
                    #or name.split('/')[-1].split('_')[1] == 'epitope':
                score = [float(line.strip().split()[0]) for line in lines]
            else:
                score = [float(line.strip().split()[2]) for line in lines]
        # with open('./compare_tools/SEMAi/272_SEMAi1D_result.txt', 'r') as re:
        # with open('./data/ind264/ind_result264.txt', 'r') as re:
            #lines = re.readlines()
            label = [float(line.strip().split()[0]) for line in lines]
        dic['label'] = label
        dic['score'] = score
        data.append(dic)
        dic = {}
        print(name)
    return data


def get_data_self(path):
    data = []
    dic = {}
    for name in path:
        with open(name, 'r') as da:
            lines = da.readlines()
            label = [float(line.strip().split()[0]) for line in lines]
            score = [float(line.strip().split()[1]) for line in lines]
        dic['label'] = label
        dic['score'] = score
        data.append(dic)
        dic = {}
        print(name)
    return data


def get_auc(alldata):
    auc_data = {}
    all_data = []
    for data in alldata:
        fp, tp, _ = sk.roc_curve(data.get('label'), data.get('score'))
        score = sk.roc_auc_score(data.get('label'), data.get('score'))
        auc_data['fp'] = fp
        auc_data['tp'] = tp
        auc_data['score'] = score
        all_data.append(auc_data)
        auc_data = {}
    return all_data


def get_ap(alldata):
    ap_data = {}
    all_data = []
    for data in alldata:
        precision, recall, _ = sk.precision_recall_curve(data.get('label'), data.get('score'))
        ap = sk.average_precision_score(data.get('label'), data.get('score'))
        ap_data['precision'] = precision
        ap_data['recall'] = recall
        ap_data['ap'] = ap
        all_data.append(ap_data)
        ap_data = {}
    return all_data


def get_one_ap(dataone, name):
    ap_data = {}
    precision, recall, wei = sk.precision_recall_curve(dataone.get('label'), dataone.get('score'), pos_label=1)
    ap = sk.average_precision_score(dataone.get('label'), dataone.get('score'))
    ap_data['precision'] = precision
    ap_data['recall'] = recall
    ap_data['ap'] = ap
    ap_data['isone'] = name
    return ap_data


def get_one_auc(dataone, name):
    auc_data = {}
    fp, tp, _ = sk.roc_curve(dataone.get('label'), dataone.get('score'))
    score = sk.roc_auc_score(dataone.get('label'), dataone.get('score'))
    auc_data['fp'] = fp
    auc_data['tp'] = tp
    auc_data['score'] = score
    auc_data['isone'] = name
    return auc_data


def mean_auc(auc_data, name):
    dic = {}
    mean_fp_list = []
    mean_fp = 0
    mean_tp_list = []
    mean_tp = 0
    mean_score = 0
    num = min(len(auc_data[i].get('fp')) for i in range(len(auc_data)))
    for i in range(len(auc_data)):
        mean_score = mean_score + auc_data[i].get('score')
    mean_score = mean_score / len(auc_data)
    for i in range(num):
        mean_fp = 0
        mean_tp = 0
        for j in range(len(auc_data)):
            mean_fp = mean_fp + auc_data[j].get('fp')[i]
            mean_tp = mean_tp + auc_data[j].get('tp')[i]
        mean_fp_list.append(mean_fp / len(auc_data))
        mean_tp_list.append(mean_tp / len(auc_data))
    dic['fp'] = mean_fp_list
    dic['tp'] = mean_tp_list
    dic['score'] = mean_score
    dic['ismean'] = name
    return dic


def mean_ap(ap_data, name):
    dic = {}
    mean_precision_list = []
    mean_precision = 0
    mean_recall_list = []
    mean_recall = 0
    mean_ap = 0
    num = min(len(ap_data[i].get('precision')) for i in range(len(ap_data)))
    for i in range(len(ap_data)):
        mean_ap = mean_ap + ap_data[i].get('ap')
    mean_ap = mean_ap / len(ap_data)
    for i in range(num):
        mean_precision = 0
        mean_recall = 0
        for j in range(len(ap_data)):
            mean_precision = mean_precision + ap_data[j].get('precision')[i]
            mean_recall = mean_recall + ap_data[j].get('recall')[i]
        mean_precision_list.append(mean_precision / len(ap_data))
        mean_recall_list.append(mean_recall / len(ap_data))
    dic['precision'] = mean_precision_list
    dic['recall'] = mean_recall_list
    dic['ap'] = mean_ap
    dic['ismean'] = name
    return dic


def show_roc(roc_data, path):
    plt.figure(dpi=300)
    for data, num in zip(roc_data, range(len(roc_data))):
        # label = str(num + 1) + ' AUC: '
        if data.get('ismean'):
            label = data.get('ismean') + ' mean AUC: '
        if data.get('isone'):
            label = data.get('isone')
        if data.get('isone') == 'DLBEpitope' or data.get('isone') == 'epitope3D':
            x, y = data.get('fp')[1], data.get('tp')[1]
            plt.scatter(x, y, color=color_list[num], lw=1,
                 label=label)
        else:
            label = label + ' (AUC:'
            plt.plot(data.get('fp'), data.get('tp'), color=color_list[num], lw=1,
                 label=label + str('%.3f' % data.get('score')) + ')')
    fontsize = 14
    plt.xlabel('False Postive Rate', fontsize=fontsize)
    plt.ylabel('True Postive Rate', fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(path)
    plt.clf()
    # plt.show()


def show_pr(ap_data, path):
    plt.figure(dpi=300)
    for data, num in zip(ap_data, range(len(ap_data))):
        # label = str(num + 1) + ' AP: '
        if data.get('ismean'):
            label = data.get('ismean') + ' mean AP: '
        if data.get('isone'):
            label = data.get('isone')
        if data.get('isone') == 'DLBEpitope' or data.get('isone') == 'epitope3D':
            x, y = data.get('recall')[1], data.get('precision')[1]
            plt.scatter(x, y, color=color_list[num], lw=1,
                 label=label)
        else:
            label = label + ' (AP:'
            plt.plot(data.get('recall'), data.get('precision'), color=color_list[num], lw=1,
                 label=label + str('%.3f' % data.get('ap')) + ')')
    fontsize = 14
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(path)
    plt.clf()
    # plt.show()


if __name__ == '__main__':
    #  b' ：蓝色
    # 'm' ：洋红色
    # 'g' ：绿色
    # 'y' ：黄色
    # 'r' ：红色
    # 'k' ：黑色
    # 'c' ：青绿色
    #  按顺序填颜色
    color_list = ['b', 'm', 'r', 'y', 'k', 'g', 'c']

    # name_list = []
    # alldata = os.listdir('./data/pt')
    # for name in alldata:
    #     name_list.append('./data/pt/' + name)
    # data_pt = get_data(name_list)
    # #
    # name_list = []
    # alldata = os.listdir('./data/yy')
    # for name in alldata:
    #     name_list.append('./data/yy/' + name)
    # data_yy = get_data(name_list)
    # name_list = []
    # alldata = os.listdir('./data/du_old')
    # for name in alldata:
    #     name_list.append('./data/du_old/' + name)
    # data_odu = get_data(name_list)
    # name_list = []
    # alldata = os.listdir('./data/du')
    # for name in alldata:
    #     name_list.append('./data/du/' + name)
    # data_du = get_data(name_list)
    roc_list = []
    ap_list = []

    # name_list.append('./data/zy_result_new_du.txt')
    # data = get_data(name_list)
    #
    # roc_list.append(get_one_auc(data[-1], 'transform down/up'))
    # ap_list.append(get_one_ap(data[-1], 'transform down/up'))
    # name_list = []
    # name_list.append('./data/zy_result_du_ppf.txt')
    # data = get_data(name_list)
    #
    # roc_list.append(get_one_auc(data[0], 'ppf down/up'))
    # ap_list.append(get_one_ap(data[0], 'ppf down/up'))
    # name_list = []
    # name_list.append('./data/zy_result_1_du.txt')
    # odu_data = get_data(name_list)
    # roc_list.append(get_one_auc(odu_data[0], 'new 1.2down/up transform'))
    # ap_list.append(get_one_ap(odu_data[0], 'new 1.2down/up transform'))
    #
    # name_list = []
    # name_list.append('./data/zy_result_1_du_1.txt')
    # odu_data = get_data(name_list)
    # roc_list.append(get_one_auc(odu_data[0], 'new 1down/up transform'))
    # ap_list.append(get_one_ap(odu_data[0], 'new 1down/up transform'))

    name_list = []
    name_list.append('./compare_tools/SEMAi/272_SEMAi3D_result.txt')
    # name_list.append('./data/ind264/ind_bepipred_label264.txt')
    odu_data = get_data(name_list)
    roc_list.append(get_one_auc(odu_data[0], 'SEMAi-3D'))
    ap_list.append(get_one_ap(odu_data[0], 'SEMAi-3D'))

    name_list = []
    name_list.append('./compare_tools/discotope3/272_discotope3_result.txt')
    # name_list.append('./data/ind264/ind_discotope_label264.txt')
    odu_data = get_data(name_list)
    roc_list.append(get_one_auc(odu_data[0], 'Discotope3'))
    ap_list.append(get_one_ap(odu_data[0], 'Discotope3'))

    name_list = []
    name_list.append('./compare_tools/SEMAi/272_SEMAi1D_result.txt')
    # name_list.append('./data/ind264/ind_DLBEpitope_label264.txt')
    odu_data = get_data(name_list)
    roc_list.append(get_one_auc(odu_data[0], 'SEMAi-1D'))
    ap_list.append(get_one_ap(odu_data[0], 'SEMAi-1D'))

    name_list = []
    name_list.append('./compare_tools/seppa3/272_seppa3_result.txt')
    # name_list.append('./data/ind264/ind_seppa_label264.txt')
    odu_data = get_data(name_list)
    roc_list.append(get_one_auc(odu_data[0], 'SEPPA 3.0'))
    ap_list.append(get_one_ap(odu_data[0], 'SEPPA 3.0'))

    """name_list = []
    name_list.append('./compare_tools/epidope3D/272_epidope3D_result.txt')
    # name_list.append('./data/ind264/ind_epitope_label264.txt')
    odu_data = get_data(name_list)
    roc_list.append(get_one_auc(odu_data[0], 'epitope3D'))
    ap_list.append(get_one_ap(odu_data[0], 'epitope3D'))"""

    name_list = []
    name_list.append('./compare_tools/ellipro/272_ellipro_result.txt')
    # name_list.append('./data/ind264/ind_epitope_label264.txt')
    odu_data = get_data(name_list)
    roc_list.append(get_one_auc(odu_data[0], 'ellipro'))
    ap_list.append(get_one_ap(odu_data[0], 'ellipro'))

    name_list = []
    name_list.append('272_ln2_result_(0).txt')
    # name_list.append('data/ind5/zy_ind_1.txt')
    # name_list.append('./data/ntest/zy724_result4_pt.txt')
    # name_list.append('./data/ind264/zy_result_ind264.txt')
    odu_data = get_data_self(name_list)
    roc_list.append(get_one_auc(odu_data[0], 'PCBEP'))
    ap_list.append(get_one_ap(odu_data[0], 'PCBEP'))

    # roc_data = get_auc(data_du)
    # ap_data = get_ap(data_du)
    #
    # roc_dic_du = mean_auc(roc_data, 'down/up two transform')
    # ap_dic_du = mean_ap(ap_data, 'down/up two transform')
    # roc_list.append(roc_dic_du)
    # ap_list.append(ap_dic_du)
    #
    # roc_data = get_auc(data_odu)
    # ap_data = get_ap(data_odu)
    #
    # roc_dic_odu = mean_auc(roc_data, 'down/up one transform')
    # ap_dic_odu = mean_ap(ap_data, 'down/up one transform')
    # roc_list.append(roc_dic_odu)
    # ap_list.append(ap_dic_odu)
    #
    # roc_data = get_auc(data_pt)
    # ap_data = get_ap(data_pt)
    #
    # roc_dic = mean_auc(roc_data, 'transform')
    # ap_dic = mean_ap(ap_data, 'transform')
    # roc_list.append(roc_dic)
    # ap_list.append(ap_dic)
    # #
    # roc_data_yy = get_auc(data_yy)
    # ap_data_yy = get_ap(data_yy)
    #
    # roc_dic_yy = mean_auc(roc_data_yy, 'baseline')
    # ap_dic_yy = mean_ap(ap_data_yy, 'baseline')
    # roc_list.append(roc_dic_yy)
    # ap_list.append(ap_dic_yy)

    # roc_list = []
    # ap_list = []
    # roc_list.append(roc_dic)
    # roc_list.append(roc_dic_yy)
    # # roc_list.append(roc_data[0])
    # ap_list.append(ap_dic)
    # ap_list.append(ap_dic_yy)
    # ap_list.append(ap_data[0])

    auc_picture_path = './picture/ROC_all.png'
    ap_picture_path = './picture/pr_all.png'
    # auc_picture_path = './picture3/ROC_nind_all.png'`
    # ap_picture_path = './picture3/pr_nind_all.png'
    # show_roc(roc_list, auc_picture_path)
    # show_pr(ap_list, ap_picture_path)
    show_roc(roc_list, auc_picture_path)
    show_pr(ap_list, ap_picture_path)
