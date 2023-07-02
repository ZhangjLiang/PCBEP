#AP AUC of PCBEP from 5 fold
import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sk


def get_data(path):
    with open(path, 'r') as da:
        lines = da.readlines()
        label = [float(line.strip().split()[0]) for line in lines]
        score = [float(line.strip().split()[1]) for line in lines]
    return label, score


def show_roc(fp, tp, t_fp, t_tp, score, t_score, path):
    plt.figure(1)
    plt.plot(t_fp, t_tp, color='black', lw=1, label='transformer AUC: ' + str(round(t_score, 4)))
    plt.plot(fp, tp, color='red', lw=1, label='baseline AUC: ' + str(round(score, 4)))
    fontsize = 14
    plt.xlabel('False Postive Rate', fontsize=fontsize)
    plt.ylabel('True Postive Rate', fontsize=fontsize)
    plt.title('ROC')
    plt.legend()
    plt.savefig('./picture/ROC_5.png')
    plt.clf()
    # plt.show()


def show_pr(t_precision, t_recall, precision, recall, t_ap, ap, path):
    plt.figure(1)
    plt.plot(t_recall, t_precision, color='black', lw=1, label='transformer AP: ' + str(round(t_ap, 4)))
    plt.plot(recall, precision, color='red', lw=1, label='baseline AP: ' + str(round(ap, 4)))
    fontsize = 14
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('./picture/pr_5.png')
    plt.clf()
    # plt.show()


if __name__ == '__main__':
    tnet_label1, tnet_score1 = get_data('./272_ln2_result_(0).txt')
    tnet_label2, tnet_score2 = get_data('./272_ln2_result_(1).txt')
    tnet_label3, tnet_score3 = get_data('./272_ln2_result_(2).txt')
    tnet_label4, tnet_score4 = get_data('./272_ln2_result_(3).txt')
    tnet_label5, tnet_score5 = get_data('./272_ln2_result_(4).txt')
    # tnet_label_m, tnet_score_m = get_data('data/test/zy_result_test.txt')
    # tnet_score = [(tnet_score1[i] + tnet_score2[i] + tnet_score3[i] + tnet_score4[i] + tnet_score5[i]) / 5 for i in range(len(tnet_score5))]

    # tnet_label1, tnet_score1 = get_data('data/du_old/zy_result_du_0.txt')
    # tnet_label2, tnet_score2 = get_data('data/du/zy_result_1_du.txt')
    # tnet_label3, tnet_score3 = get_data('data/du/zy_result_2_du.txt')
    # tnet_label4, tnet_score4 = get_data('data/du/zy_result_3_du.txt')
    # tnet_label5, tnet_score5 = get_data('data/du/zy_result_4_du.txt')

    # tnet_label1, tnet_score1 = get_data('data/npt3/zy724_result0_pt.txt')
    # tnet_label2, tnet_score2 = get_data('data/npt3/zy724_result1_pt.txt')
    # tnet_label3, tnet_score3 = get_data('data/npt3/zy724_result2_pt.txt')
    # tnet_label4, tnet_score4 = get_data('data/npt3/zy724_result3_pt.txt')
    # tnet_label5, tnet_score5 = get_data('data/npt3/zy724_result4_pt.txt')

    t_precision1, t_recall1, _ = sk.precision_recall_curve(tnet_label1, tnet_score1)
    t_precision2, t_recall2, _ = sk.precision_recall_curve(tnet_label2, tnet_score2)
    t_precision3, t_recall3, _ = sk.precision_recall_curve(tnet_label3, tnet_score3)
    t_precision4, t_recall4, _ = sk.precision_recall_curve(tnet_label4, tnet_score4)
    t_precision5, t_recall5, _ = sk.precision_recall_curve(tnet_label5, tnet_score5)
    # t_precision_m, t_recall_m, _ = sk.precision_recall_curve(tnet_label_m, tnet_score_m)
    # t_precision_m, t_recall_m, _ = sk.precision_recall_curve(tnet_label5, tnet_score)
    # precision, recall, _ = sk.precision_recall_curve(label, score)
    t_ap1 = sk.average_precision_score(tnet_label1, tnet_score1)
    t_ap2 = sk.average_precision_score(tnet_label2, tnet_score2)
    t_ap3 = sk.average_precision_score(tnet_label3, tnet_score3)
    t_ap4 = sk.average_precision_score(tnet_label4, tnet_score4)
    t_ap5 = sk.average_precision_score(tnet_label5, tnet_score5)
    # t_ap_m = sk.average_precision_score(tnet_label_m, tnet_score_m)
    # t_ap_m = sk.average_precision_score(tnet_label5, tnet_score)
    # ap = sk.average_precision_score(label, score)

    t_fp1, t_tp1, _ = sk.roc_curve(tnet_label1, tnet_score1)
    t_fp2, t_tp2, _ = sk.roc_curve(tnet_label2, tnet_score2)
    t_fp3, t_tp3, _ = sk.roc_curve(tnet_label3, tnet_score3)
    t_fp4, t_tp4, _ = sk.roc_curve(tnet_label4, tnet_score4)
    t_fp5, t_tp5, _ = sk.roc_curve(tnet_label5, tnet_score5)
    # t_fp_m, t_tp_m, _ = sk.roc_curve(tnet_label_m, tnet_score_m)
    # t_fp_m, t_tp_m, _ = sk.roc_curve(tnet_label5, tnet_score)
    # fp, tp, _ = sk.roc_curve(label, score)
    # roc_score = sk.roc_auc_score(label, score)
    t_score1 = sk.roc_auc_score(tnet_label1, tnet_score1)
    t_score2 = sk.roc_auc_score(tnet_label2, tnet_score2)
    t_score3 = sk.roc_auc_score(tnet_label3, tnet_score3)
    t_score4 = sk.roc_auc_score(tnet_label4, tnet_score4)
    t_score5 = sk.roc_auc_score(tnet_label5, tnet_score5)
    # t_score_m = sk.roc_auc_score(tnet_label_m, tnet_score_m)
    # t_score_m = sk.roc_auc_score(tnet_label5, tnet_score)

    t_fp, t_tp, t_score = [], [], []
    t_recall, t_precision, t_ap = [], [], []
    for i in range(min(len(t_fp1), len(t_fp2),len(t_fp3),len(t_fp4),len(t_fp5))):
        t_fp.append((t_fp1[i] + t_fp2[i] + t_fp3[i] + t_fp4[i] + t_fp5[i]) / 5)
        t_tp.append((t_fp1[i] + t_tp2[i] + t_tp3[i] + t_tp4[i] + t_tp5[i]) / 5)
    for i in range(min(len(t_recall1), len(t_recall2),len(t_recall3),len(t_recall4),len(t_recall5))):
        t_recall.append((t_recall1[i] + t_recall2[i] + t_recall3[i] + t_recall4[i] + t_recall5[i]) / 5)
        t_precision.append(
            (t_precision1[i] + t_precision2[i] + t_precision3[i] + t_precision4[i] + t_precision5[i]) / 5)
    t_score = (t_score1 + t_score2 + t_score3 + t_score4 + t_score5) / 5
    t_ap = (t_ap1 + t_ap2 + t_ap3 + t_ap4 + t_ap5) / 5

    # show_roc(fp, tp, t_fp, t_tp, roc_score, t_score)
    # show_pr(t_precision, t_recall, precision, recall, t_ap, ap)

    plt.figure(dpi=300)
    plt.plot(t_fp1, t_tp1, color='g', lw=1, label='1-Fold Cross Validation (AUC:' + str('%.3f' % t_score1) + ')')
    plt.plot(t_fp2, t_tp2, color='r', lw=1, label='2-Fold Cross Validation (AUC:' + str('%.3f' % t_score2) + ')')
    plt.plot(t_fp3, t_tp3, color='y', lw=1, label='3-Fold Cross Validation (AUC:' + str('%.3f' % t_score3) + ')')
    plt.plot(t_fp4, t_tp4, color='b', lw=1, label='4-Fold Cross Validation (AUC:' + str('%.3f' % t_score4) + ')')
    plt.plot(t_fp5, t_tp5, color='c', lw=1, label='5-Fold Cross Validation (AUC:' + str('%.3f' % t_score5) + ')')
    # plt.plot(t_fp_m, t_tp_m, color='black', lw=1, label='Mean-Fold Cross Validation (AUC:' + str('%.3f' % t_score_m) + ')')
    fontsize = 14
    plt.xlabel('False Postive Rate', fontsize=fontsize)
    plt.ylabel('True Postive Rate', fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('./picture/272_ROC_5.png')
    plt.clf()

    plt.figure(dpi=300)
    plt.plot(t_recall1, t_precision1, color='g', lw=1, label='1-Fold Cross Validation (AP:' + str('%.3f' % t_ap1) + ')')
    plt.plot(t_recall2, t_precision2, color='r', lw=1, label='2-Fold Cross Validation (AP:' + str('%.3f' % t_ap2) + ')')
    plt.plot(t_recall3, t_precision3, color='y', lw=1, label='3-Fold Cross Validation (AP:' + str('%.3f' % t_ap3) + ')')
    plt.plot(t_recall4, t_precision4, color='b', lw=1, label='4-Fold Cross Validation (AP:' + str('%.3f' % t_ap4) + ')')
    plt.plot(t_recall5, t_precision5, color='c', lw=1, label='5-Fold Cross Validation (AP:' + str('%.3f' % t_ap5) + ')')
    # plt.plot(t_recall_m, t_precision_m, color='black', lw=1, label='Mean-Fold Cross Validation (AP:' + str('%.3f' % t_ap_m) + ')')
    fontsize = 14
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('./picture/272_pr_5.png')
    plt.clf()
