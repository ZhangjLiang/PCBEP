from itertools import product
from random import random
from turtle import shape

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ones
from torch_geometric.data import Data, DataLoader
import numpy as np
from torch.nn import Sequential as Seq, Dropout, GELU, Linear as Lin, ReLU, BatchNorm1d as BN, LayerNorm as LN, Softmax
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, radius, global_mean_pool, knn
from torch_geometric.data import batch
from sklearn import metrics
from zy_pytorchtools import EarlyStopping
from divide_tools import divide_cdhit
# from torch_geometric.utils import dropout_adj, false_negative, false_positive, recall, f1_score, true_negative, \
# precision, true_positive, to_dense_batch
from torch.autograd import Variable
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor, Adj
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Callable, Union
from torch import Tensor
from PPFConv import PPFConv
from PointTransformerConv import PointTransformerConv
import os
import sys
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.spatial
from torch_scatter import scatter_add

Center = T.Center()
Normalscale = T.NormalizeScale()
Delaunay = T.Delaunay()
Normal = T.GenerateMeshNormals()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

def normalize_point_pos(pos):
    # pos_AB=torch.cat([pos_A, pos_B])
    pos = pos - pos.mean(dim=-2, keepdim=True)
    # pos_B=pos_B-pos_AB.mean(dim=-2, keepdim=True)
    scale = (1 / pos.abs().max()) * 0.999999
    pos = pos * scale
    # scale_B = (1 / pos_B.abs().max()) * 0.999999
    # pos_B = pos_B * scale_B
    return pos


def load_data(data_path):
    print('loading data')
    data_list = []

    with open(data_path, 'r') as f:
        n_g = int(f.readline().strip())
        num = 0
        count = 0
        for i in range(n_g):  # for each protein
            n = int(f.readline().strip())  # atom number
            point_tag = []
            point_fea_pssm = []
            point_pos = []
            point_aa = []
            point_esm = []
            aa_y = []
            mask = []

            # print(count)
            '''fr = open('./esm_result/724_' + str(count) + '.txt', 'r')
            li = fr.readlines()
            esm_list = [[] * num for num in range(len(li))]
            for x in range(len(li)):
                new_list = li[x].split()
                for y in range(len(new_list)):
                    esm_list[x].append(float(new_list[y]))

            esm_list = torch.tensor(esm_list, dtype=torch.float)'''
            for j in range(n):
                row = f.readline().strip().split()
                point_tag.append(int(row[1]))
                mask.append(int(row[2]))
                pos, fea_pssm = np.array([float(w) for w in row[3:6]]), np.array([float(w) for w in row[6:]])
                point_pos.append(pos)
                point_fea_pssm.append(fea_pssm)
                point_aa.append(int(row[0]))

            flag = -1
            for i in range(len(point_aa)):
                if (flag != point_aa[i]):
                    flag = point_aa[i]
                    aa_y.append(point_tag[i])
            # print(aa_y)
            x = torch.tensor(point_fea_pssm, dtype=torch.float)  # 39
            y = torch.tensor(point_tag)
            pos = torch.tensor(point_pos, dtype=torch.float)  # 3
            mask = torch.tensor(mask)

            fea_esm = torch.tensor(point_esm, dtype=torch.float)

            # pos=normalize_point_pos(pos)
            data = Data(x=x, y=y, pos=pos)
            # print(data.norm)

            for i in range(len(point_aa)):
                point_aa[i] = point_aa[i] + num
            num = num + len(aa_y)

            aa = torch.tensor(point_aa)
            # print(aa)
            number = len(aa_y)
            aa_y = torch.tensor(aa_y)

            data.aa = aa
            data.aa_y = aa_y
            data.num = number
            data.mask = mask
            #data.esm_list = esm_list

            data = Center(data)
            # data = Normalscale(data)
            data = Delaunay(data)
            data = Normal(data)

            data = data.to(device)
            # if count == 289:
            #     count += 1
            #     continue
            data_list.append(data)
            count += 1
    # print(data_list)
    # torch.save(data_list, '724_dataset.pt')
    return data_list


def load_InDdata(data_path, datasetnum):
    print('loading data')
    data_list = []

    # pass_list = [16,25,26,31,32,39]
    with open(data_path, 'r') as f:
        n_g = int(f.readline().strip())
        num = 0
        count = 0
        for i in range(n_g):  # for each protein
            # if count in pass_list:
            #     continue
            n = int(f.readline().strip())  # atom number
            point_tag = []
            point_fea_pssm = []
            point_pos = []
            point_aa = []
            point_esm = []
            aa_y = []
            mask = []
            # print(count)
            fr = open('/users/PCON0022/yuy702/zy/esm/esm_result68/' + str(datasetnum) + '_' + str(count) + '.txt', 'r')
            li = fr.readlines()
            esm_list = [[] * num for num in range(len(li))]
            for x in range(len(li)):
                new_list = li[x].split()
                for y in range(len(new_list)):
                    esm_list[x].append(float(new_list[y]))

            esm_list = torch.tensor(esm_list, dtype=torch.float)
            for j in range(n):
                row = f.readline().strip().split()
                point_tag.append(int(row[1]))
                mask.append(int(row[2]))
                pos, fea_pssm = np.array([float(w) for w in row[3:6]]), np.array([float(w) for w in row[6:]])
                point_pos.append(pos)
                point_fea_pssm.append(fea_pssm)
                point_aa.append(int(row[0]))

            flag = -1
            for i in range(len(point_aa)):
                if (flag != point_aa[i]):
                    flag = point_aa[i]
                    aa_y.append(point_tag[i])
            # print(aa_y)
            x = torch.tensor(point_fea_pssm, dtype=torch.float)  # 39
            y = torch.tensor(point_tag)
            pos = torch.tensor(point_pos, dtype=torch.float)  # 3
            mask = torch.tensor(mask)

            fea_esm = torch.tensor(point_esm, dtype=torch.float)

            # pos=normalize_point_pos(pos)
            data = Data(x=x, y=y, pos=pos)
            # print(data.norm)

            for i in range(len(point_aa)):
                point_aa[i] = point_aa[i] + num
            num = num + len(aa_y)

            aa = torch.tensor(point_aa)
            # print(aa)
            number = len(aa_y)
            aa_y = torch.tensor(aa_y)

            data.aa = aa
            data.aa_y = aa_y
            data.num = number
            data.mask = mask
            data.esm_list = esm_list
            data = Center(data)
            # data = Normalscale(data)
            data = Delaunay(data)
            data = Normal(data)
            data = data.to(device)
            data_list.append(data)
            count += 1
    # torch.save(data_list, '' + str(datasetnum) + '_dataset.pt')
    return data_list


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU(), Dropout(0.3))
        for i in range(1, len(channels))
    ])


#
# def local_nn(channels):
#     return Seq(*[
#         Seq(Lin(channels[i - 1], channels[i]))
#         for i in range(1, len(channels))
#     ])
#
#
# def global_nn(channels):
#     return Seq(*[
#         Seq(BN(channels), ReLU())
#     ])


def generate_normal(pos, batch):
    data_norm = []
    batch_list = torch.unique(batch)
    for b in batch_list:
        pos_temp = pos[batch == b]
        pos_temp = pos_temp - pos_temp.mean(dim=-2, keepdim=True)
        pos_temp = pos_temp.cpu().numpy()
        tri = scipy.spatial.Delaunay(pos_temp, qhull_options='QJ')
        face = torch.from_numpy(tri.simplices)

        data_face = face.t().contiguous().to(device, torch.long)
        pos_temp = torch.tensor(pos_temp).to(device)

        vec1 = pos_temp[data_face[1]] - pos_temp[data_face[0]]
        vec2 = pos_temp[data_face[2]] - pos_temp[data_face[0]]
        face_norm = F.normalize(vec1.cross(vec2), p=2, dim=-1)  # [F, 3]

        idx = torch.cat([data_face[0], data_face[1], data_face[2]], dim=0)
        face_norm = face_norm.repeat(3, 1)

        norm = scatter_add(face_norm, idx, dim=0, dim_size=pos_temp.size(0))
        norm = F.normalize(norm, p=2, dim=-1)  # [N, 3]

        data_norm.append(norm)

    return torch.cat(data_norm, dim=0)


# class PPFConv1(torch.nn.Module):
#     def __init__(self, k, local_nn, global_nn):
#         super(PPFConv1, self).__init__()
#         self.k = k
#         self.r = None
#         self.conv = PPFConv(local_nn, global_nn, add_self_loops=True)
#
#     def forward(self, x, pos, normal, batch):
#         row, col = knn(pos, pos, self.k, batch, batch)
#         # row, col = radius(pos, pos, self.r, batch, batch, max_num_neighbors=64)
#         edge_index = torch.stack([col, row], dim=0)
#         x = self.conv(x, pos, normal, edge_index, self.r)
#         # x = self.conv(x, pos, normal, edge_index)
#         return x


class PointTransformerConv1(torch.nn.Module):
    def __init__(self, r, in_channels, out_channels):
        super(PointTransformerConv1, self).__init__()
        self.k = None
        self.r = r
        self.pos_nn = MLP([6, out_channels])

        self.attn_nn = MLP([out_channels, out_channels])

        self.conv = PointTransformerConv(in_channels, out_channels,
                                         pos_nn=self.pos_nn,
                                         attn_nn=self.attn_nn)

    def forward(self, x, pos, normal, batch):
        # row, col = knn(pos, pos, self.k, batch, batch)
        row, col = radius(pos, pos, self.r, batch, batch, max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, pos, edge_index, normal, self.r)
        return x


class Net(torch.nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.conv1 = PointTransformerConv1(5, in_channels=59, out_channels=128)
        self.conv2 = PointTransformerConv1(8.5, in_channels=59, out_channels=128)
        self.conv3 = PointTransformerConv1(10, in_channels=59, out_channels=128)
        # self.conv4 = PointTransformerConv1(20, in_channels=39 + 20, out_channels=128)

        # self.neck = Seq(Lin(128 + 128 + 128 , 512), BN(512), ReLU(), Dropout(0.3))
        self.neck = Seq(Lin(128 + 128 + 128, 1280), LN(1280), GELU())
        self.neck_esm = Seq(Lin(1280 + 1280, 1280), LN(1280), GELU())
        # self.conv5 = PointTransformerConv1(15, in_channels=512, out_channels=512)
        self.conv5 = PointTransformerConv1(15, in_channels=1280, out_channels=1280)
        # self.mlp = Seq(Lin(512, 256), BN(256), ReLU(), Dropout(0.3), Lin(256, out_channels))
        # self.mlp = Seq(Lin(516, 256), ReLU(), Lin(256, out_channels))
        # self.mlp_esm = Seq(Lin(1280, 512), BN(512), ReLU(), Dropout(0.3), Lin(512, 256), Lin(256, 1)) # 150 120 45
        self.mlp_esm = Seq(Lin(1280, 150), LN(150), GELU(), Lin(150, 120), LN(120), GELU(), Lin(120, 45), LN(45),
                           GELU(), Lin(45, 1))
        self.mlp_only = Seq(Lin(1280, 2))
        self.soft_max = Softmax(dim=1)

    def forward(self, data, use=None):
        x0, pos, batch, normal, pool_batch, aa_num, mask, esm_list = data.x, data.pos, data.batch, data.norm, data.aa, data.num, data.mask, data.esm_list

        # atom to residue
        flag = torch.Tensor([-1]).to(device)
        num = -1
        for i in range(len(pool_batch)):
            if not torch.eq(pool_batch[i], flag):
                flag = pool_batch[i].clone()
                num = num + 1
                pool_batch[i] = torch.Tensor([num]).to(device)
            else:
                pool_batch[i] = torch.Tensor([num]).to(device)

        x1 = self.conv1(x0, pos, normal, batch)
        x2 = self.conv2(x0, pos, normal, batch)
        x3 = self.conv3(x0, pos, normal, batch)
        # x4 = self.conv4(x0, pos, normal, batch)
        # print(batch)
        out = self.neck(torch.cat([x1, x2, x3], dim=1))
        out = global_mean_pool(out, pool_batch)  # out-512
        # out = global_max_pool(pool_batch)
        # print(out)

        # residual batch
        # print(num)
        num_total = 0
        for i in range(len(aa_num)):
            num_total += aa_num[i]
        # print(num_total)
        # 一个全为0的tensor
        aa_batch = torch.zeros(num_total).to(device)
        number = 0
        for m in range(len(aa_num)):
            # print(m)
            for n in range(aa_num[m].item()):
                aa_batch[n + number] = m
            number += aa_num[m].item()
        aa_batch = aa_batch.long()
        aa_pos = global_mean_pool(pos, pool_batch)
        aa_norm = generate_normal(aa_pos, aa_batch).to(device)
        # out = self.conv5(aa_pos, aa_norm, aa_batch)
        # out_esm = self.mlp_esm(esm_list)
        # out = out + out_esm
        # out = out + esm_list
        out = self.conv5(out, aa_pos, aa_norm, aa_batch)
        # out = self.neck_conv5(out)
        # out = self.mlp(out)
        mask = global_max_pool(mask.unsqueeze(dim=1), pool_batch).squeeze()
        mask = mask == 1
        # out = out[mask]
        # data.label = data.aa_y
        out_esm = self.mlp_esm(out + esm_list)
        out_only = self.mlp_only(esm_list)
        out_only = self.soft_max(out_only)
        # out_esm = self.soft_max(out_esm)
        # out_esm = out_esm[mask]
        # out_only = out_only[mask]
        # out = (out_esm[:, 1].unsqueeze(1) + out_only[:, 1].unsqueeze(1))
        out = out_esm + out_only[:, 1].unsqueeze(1)
        # out = self.soft_max(out_esm + out_only)
        # out = out_esm + out_only
        # out = out[:, 1].unsqueeze(1)
        if use == 'test':
            out = F.sigmoid(out)
            out = mask.unsqueeze(dim=1) * out
            data.label = data.aa_y
        else:
            out = out[mask]
            data.label = data.aa_y[mask]
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, p_weight):
        pos_weight = torch.FloatTensor([p_weight]).to(device)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction='none')
        # BCE_loss = F.binary_cross_entropy(inputs, targets)
        # pt = torch.exp(-BCE_loss)
        # F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        # return F_loss.mean()
        return BCE_loss.mean()


def train_model(model, patience, n_epochs, checkpoint, p_weight):
    train_losses = []
    valid_losses = []
    label_total = []
    score_total = []
    avg_train_losses = []
    avg_valid_losses = []
    # check_point='checkpoint_knn.pt'
    early_stopping = EarlyStopping(patience=patience, path=checkpoint, verbose=True)
    # pos_weight = torch.FloatTensor([10.0]).to(device)

    for epoch in range(1, n_epochs + 1):

        model.train()
        for data in trainloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            # label = data.aa_y.float()
            label = data.label.float()
            label = label.unsqueeze(1)
            loss = focalloss(out, label, p_weight)
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for data in valloader:
                data = data.to(device)
                out = model(data)
                score = torch.sigmoid(out)
                score_total.extend(score.detach().cpu().numpy())
                # print(score_total)
                label = data.label.float()
                # label = data.aa_y.float()
                label = label.unsqueeze(1)
                label_total.extend(label.detach().cpu().numpy())
                loss = focalloss(out, label, p_weight)

                # scheduler.step(loss)
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        auc = metrics.roc_auc_score(label_total, score_total)
        ap = metrics.average_precision_score(label_total, score_total)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'AUC: {auc:.5f}' +
                     f'AP: {ap:.5f}')

        print(print_msg)

        train_losses = []
        valid_losses = []
        label_total = []
        score_total = []

        scheduler.step(valid_loss)
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # model = torch.load(checkpoint)

    return avg_train_losses, avg_valid_losses


#dataset = load_data('724_data_feature_surface.txt')
# InDdataset = load_InDdata('/users/PCON0022/yuy702/zjl/esm/64_data_feature_surface.txt',64)
InDdataset = torch.load('./272_dataset.pt')
dataset = torch.load('./724_dataset.pt')
# InDdataset = torch.load('/users/PCON0022/yuy702/zy/esm/272_dataset.pt')
train, val, test = dataset[: 505], dataset[505: 608], dataset[608:]
# train, val = dataset[: 505], dataset[505: 608]
# test = InDdataset
testloader = DataLoader(test, batch_size=1)
checkpoint_list = []
p_weight = [3.91, 4.5, 4.76, 5.14, 5.14]

for fold in [0, 1, 2, 3, 4]:
    train, val = divide_cdhit(dataset, fold)
    trainloader = DataLoader(train, batch_size=1, shuffle=True, drop_last=True)
    valloader = DataLoader(val, batch_size=1)

    model = Net()
    focalloss = FocalLoss()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)  #
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)

    n_epochs = 15000
    patience = 20
    checkpoint = 'checkpoint/pcbep_' + str(fold) + '.pt'
    checkpoint_list.append(checkpoint)
    # train model
    train_loss, valid_loss = train_model(model, patience, n_epochs, checkpoint, p_weight[fold])

for i in range(5):
    pred_total = []
    aa_total = []
    out_total = []
    model_list = []
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint_list[i]))
    model.eval()
    model_list.append(model)
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            out = model(data, 'test')
            # out = F.sigmoid(out)
            # print(F.sigmoid(out))
            out_total.extend(out.cpu().tolist())
            # print(out_total)
            pred = out.ge(0.50).float()
            pred_total.extend(pred.detach().cpu().numpy())
            aa_total.extend(data.label.detach().cpu().numpy())
            # aa_total.extend(data.aa_y.detach().cpu().numpy())

    pred_total = torch.tensor(pred_total)
    out_total = torch.tensor(out_total)
    pred_total = pred_total.squeeze()
    out_total = out_total.squeeze()

    aa_total = torch.tensor(aa_total)
    # aa_total = aa_total.unsqueeze(1)
    # print(pred_total)
    # print(aa_total)

    correct = int(pred_total.eq(aa_total).sum().item())
    # tn = true_negative(pred_total, aa_total, 1).to(torch.float)
    # tp = true_positive(pred_total, aa_total, 1).to(torch.float)
    # fn = false_negative(pred_total, aa_total, 1).to(torch.float)
    # fp = false_positive(pred_total, aa_total, 1).to(torch.float)
    tn, fp, fn, tp = confusion_matrix(aa_total, pred_total).ravel()
    print('tn' + str(tn) + 'tp' + str(tp) + 'fn' + str(fn) + 'fp' + str(fp))
    # r = recall_score(aa_total, pred_total)
    # recall = tp / (tp + fn)
    recall = metrics.recall_score(aa_total, pred_total)
    print('recall:' + str(recall))
    # sp = tn / (fp + tn)
    # print('sp:' + str(sp))
    # precision = tp / (tp + fp)
    precision = metrics.precision_score(aa_total, pred_total)
    print('precision:' + str(precision))
    # mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + sys.float_info.epsilon)
    mcc = metrics.matthews_corrcoef(aa_total, pred_total)
    print('mcc:' + str(mcc))
    # fpr, tpr, thresholds = metrics.roc_curve(aa_total, out_total)
    auc = metrics.roc_auc_score(aa_total, out_total)
    print('AUC:' + str(auc))
    ap = metrics.average_precision_score(aa_total, out_total)
    print('AP:' + str(ap))
    # out_total1 = np.rint(out_total)
    f1 = metrics.f1_score(aa_total, pred_total)
    print('f1:' + str(f1))

    # out_total = sum(out_total.tolist(), [])
    # aa_total = sum(aa_total.tolist(), [])
    # out_total = sum(out_total.tolist())
    # aa_total = sum(aa_total.tolist())
    out_total = out_total.tolist()
    aa_total = aa_total.tolist()
    with open('result/pcbep_result_(' + str(i) + ').txt', 'w') as f:
        for i in range(len(out_total)):
            f.write(str(aa_total[i]) + '\t' + str(out_total[i]) + '\n')

print('\nind data\n')
testloader = DataLoader(InDdataset, batch_size=1)
for i in range(5):
    pred_total = []
    aa_total = []
    out_total = []
    model_list = []
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint_list[i]))
    model.eval()
    model_list.append(model)
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            out = model(data, 'test')
            # out = F.sigmoid(out)
            # print(F.sigmoid(out))
            out_total.extend(out.cpu().tolist())
            # print(out_total)
            pred = out.ge(0.50).float()
            pred_total.extend(pred.detach().cpu().numpy())
            aa_total.extend(data.label.detach().cpu().numpy())
            # aa_total.extend(data.aa_y.detach().cpu().numpy())

    pred_total = torch.tensor(pred_total)
    out_total = torch.tensor(out_total)
    pred_total = pred_total.squeeze()
    out_total = out_total.squeeze()

    aa_total = torch.tensor(aa_total)
    # aa_total = aa_total.unsqueeze(1)
    # print(pred_total)
    # print(aa_total)

    correct = int(pred_total.eq(aa_total).sum().item())
    # tn = true_negative(pred_total, aa_total, 1).to(torch.float)
    # tp = true_positive(pred_total, aa_total, 1).to(torch.float)
    # fn = false_negative(pred_total, aa_total, 1).to(torch.float)
    # fp = false_positive(pred_total, aa_total, 1).to(torch.float)
    tn, fp, fn, tp = confusion_matrix(aa_total, pred_total).ravel()
    print('tn' + str(tn) + 'tp' + str(tp) + 'fn' + str(fn) + 'fp' + str(fp))
    # r = recall_score(aa_total, pred_total)
    # recall = tp / (tp + fn)
    recall = metrics.recall_score(aa_total, pred_total)
    print('recall:' + str(recall))
    # sp = tn / (fp + tn)
    # print('sp:' + str(sp))
    # precision = tp / (tp + fp)
    precision = metrics.precision_score(aa_total, pred_total)
    print('precision:' + str(precision))
    # mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + sys.float_info.epsilon)
    mcc = metrics.matthews_corrcoef(aa_total, pred_total)
    print('mcc:' + str(mcc))
    # fpr, tpr, thresholds = metrics.roc_curve(aa_total, out_total)
    auc = metrics.roc_auc_score(aa_total, out_total)
    print('AUC:' + str(auc))
    ap = metrics.average_precision_score(aa_total, out_total)
    print('AP:' + str(ap))
    # out_total1 = np.rint(out_total)
    f1 = metrics.f1_score(aa_total, pred_total)
    print('f1:' + str(f1))

    # out_total = sum(out_total.tolist(), [])
    # aa_total = sum(aa_total.tolist(), [])
    # out_total = sum(out_total.tolist())
    # aa_total = sum(aa_total.tolist())
    out_total = out_total.tolist()
    aa_total = aa_total.tolist()
    with open('result/272_result_(' + str(i) + ').txt', 'w') as f:
        for i in range(len(out_total)):
            f.write(str(aa_total[i]) + '\t' + str(out_total[i]) + '\n')
