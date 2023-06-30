import os

import xlrd  # 导入xlrd模块
import xlwt  # 导入xlwt模块
import re


def get_true_chain_list(chain_all):
    chain_list = []
    lines = chain_all.strip().split(',')
    for chain in lines:
        chain = chain.strip()
        if len(chain) > 1:
            chain_list.append(chain[-2])
        elif len(chain) == 1:
            chain_list.append(chain)
    return chain_list


def get_true_chain_str(chain_all):
    chain_str = ''
    lines = chain_all.strip().split(',')
    for chain in lines:
        chain = chain.strip()
        if len(chain) > 1:
            chain_str = chain_str + '\t' + str(chain[-2].upper())
        elif len(chain) == 1:
            chain_str = chain_str + '\t' + str(chain)
    return chain_str


def get_list_name(fasta_path):
    name_list = []
    with open(fasta_path) as f:
        for line in f:
            name_list.append(line.strip()[1:])
            f.readline()
    return name_list


def main(input_path, out_path, fasta_path):
    name_list = get_list_name(fasta_path)
    write_all = ''
    excel_ = xlrd.open_workbook(input_path)
    Table = excel_.sheet_by_index(0)
    ncols = Table.ncols
    nrows = Table.nrows
    for row in range(1, nrows):
        pdb_name = Table.cell_value(row, 0)
        ag_list, ab_list = [], []
        for i in range(1, 6):
            if Table.cell_value(row, i * 7 + 1) == '1':
                flag = get_true_chain_list(Table.cell_value(row, i * 7 + 2))
                if flag != None:
                    ag_list.append(flag)
            if Table.cell_value(row, i * 7 + 1) == '0':
                flag = get_true_chain_str(Table.cell_value(row, i * 7 + 2))
                if flag != None:
                    ab_list.append(flag)
        for nag in ag_list:
            if (pdb_name + '-' + nag[0]) in name_list:
                write_all = write_all + pdb_name + '\t' + nag[0] + '\t,'
                for nab in ab_list:
                    write_all = write_all + nab + '\t|'
                write_all = write_all[:-1] + '\n'
    # print(write_all)
    with open(out_path, 'w') as o:
        o.write(write_all)


if __name__ == '__main__':
    main('data/data.xls', 'data/ag_at.txt')
