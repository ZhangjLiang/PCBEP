import torch
import numpy as np
import os


# from networkx.generators import line


def get_H_flag(pdbfile, chain):
    with open(pdbfile, 'r') as file:
        flag = []
        for line in file:
            if line[0:4] == 'ATOM' and line[21] == chain:
                if line[12:16].strip() == 'H':
                    flag.append(0)
                else:
                    flag.append(1)
    return flag


def get_pos(pdbfile, chain):
    with open(pdbfile, 'r') as file:
        pos = []
        for line in file:
            if line[0:4] == 'ATOM' and line[21] == chain:
                pos.append([float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())])
    return pos


def get_aa_list(pdbfile, chain):
    aa_list = []
    m = ''
    flag = 0
    num = -1
    with open(pdbfile, 'r') as file:
        for line in file:
            if line[0:4] == 'ATOM' and line[21] == chain:
                if line[22:26].strip() != m and line[21] == chain:
                    num = num + 1
                    m = line[22:26].strip()
                    flag = 1
                if flag and line[21] == chain:
                    aa_list.append(num)
    return aa_list


def get_position(pdb_name, chains):
    # try:
    pos_antigen = []
    pos_antibody = []
    ab_H_flag = []
    ag_H_flag = []
    aa_list = []
    flag = 0
    # chains = ['A', ['L', 'H']]
    # if not os.path.exists('data/pdb_update/indep/'):
    #     os.mkdir('data/pdb_update/indep/')
    pdb_path = r'./data/pdb_update/' + pdb_name.upper() + '.pdb'
    with open(pdb_path, 'r') as pdb:
        m = ''
        num = -1
        try:
            for line1 in pdb:
                row = line1.strip().split()
                # print(row)
                if row[0] == 'ATOM':
                    if (len(row[2]) == 7):
                        row[8] = row[7]
                        row[7] = row[6]
                        row[6] = row[5]
                        row[5] = row[4]
                        row[4] = row[3]
                        row[3] = row[2][4:]
                        row[2] = row[2][:4]
                    if (len(row[4]) != 1):
                        row.append('0')
                        # row[11] = row[10]
                        row[8] = row[7]
                        row[7] = row[6]
                        row[6] = row[5]
                        row[5] = row[4][1:]
                        row[4] = row[4][0]
                    if (len(row[3]) != 3):
                        row[3] = row[3][-3:]
                    if len(row[6]) > 16:
                        row[8] = row[6][-8:]
                        row[7] = row[6][-16:-8]
                        row[6] = row[6][:-16]
                    if len(row[6]) > 8:
                        row[8] = row[7]
                        row[7] = row[6][-8:]
                        row[6] = row[6][:-8]
                    if len(row[7]) > 8:
                        row[8] = row[7][-8:]
                        row[7] = row[7][:-8]
                    if row[4][0] == chains[0]:
                        pos = np.array([float(w) for w in row[6:9]])
                        pos_antigen.append(pos)
                        if line1[12:16].strip() == 'H':
                            ag_H_flag.append(0)
                        else:
                            ag_H_flag.append(1)
                    if row[4][0] in chains[1]:
                        pos = np.array([float(w) for w in row[6:9]])
                        pos_antibody.append(pos)
                        if line1[12:16].strip() == 'H':
                            ab_H_flag.append(0)
                        else:
                            ab_H_flag.append(1)
                    if str(m) != row[5] and row[4][0] == chains[0]:
                        num = num + 1
                        m = row[5]
                        flag = 1
                    if flag and row[4] == chains[0]:
                        aa_list.append(num)
        except:
            pos_antigen = get_pos(pdb_path, chains[0])
            pos_antibody = get_pos(pdb_path, chains[1])
            ag_H_flag = get_H_flag(pdb_path, chains[0])
            ab_H_flag = get_H_flag(pdb_path, chains[1])
            aa_list = get_aa_list(pdb_path, chains[0])

    pos_A = torch.tensor(pos_antigen, dtype=torch.float)
    pos_B = torch.tensor(pos_antibody, dtype=torch.float)

    ab_H_flag = torch.tensor(ab_H_flag)
    ag_H_flag = torch.tensor(ag_H_flag)

    pos_A = pos_A[ag_H_flag == 1]
    pos_B = pos_B[ab_H_flag == 1]

    aa_list = torch.tensor(aa_list)[ag_H_flag == 1]

    interface_AB = (torch.cdist(pos_A, pos_B) < 4).nonzero()  # 符合条件得下标
    interface_A_index = sorted(set(interface_AB[:, 0].tolist()))
    # interface_B_index = sorted(set(interface_AB[:, 1].tolist()))
    index = []
    for i in range(len(interface_A_index)):
        index.append(aa_list[interface_A_index[i]])
        # print(i)
    indexA = sorted(set(index))
    # print(indexA)
    return indexA


def main(input_path, output_path):
    fw = open(output_path, 'w')
    with open(input_path, 'r') as f:
        for line in f:
            name = line.strip().split(',')[0][:4]
            antigen = line.strip().split(',')[0][-2]
            antibodys = line.strip().split(',')[1].strip().split('|')
            fw.write(str(name) + '-' + str(antigen))
            at = '0'
            for i in range(len(antibodys)):
                max = 0
                antibodys[i] = antibodys[i].replace('\t', '')
                for j in range(len(antibodys[i])):
                    num = get_position(name, [antigen, antibodys[i][j]])
                    if len(num) > max:
                        max = len(num)
                        at = antibodys[i][j]
                if max == 0:
                    at = antibodys[i][j]
                fw.write(' ' + str(at))
            fw.write('\n')


if __name__ == "__main__":
    main('./data/ag_at.txt', './data/test.txt')
