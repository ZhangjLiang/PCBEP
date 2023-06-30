import os


def remove_HETATM(opath, npath):  # 输入pdb的路径，输出pdb的路径
    with open(opath, 'r') as op:
        write_all = ''
        for line in op:
            if line.startswith('<!DOCTYPE HTML PUBLIC'):
                return 0
            if line[:6] != 'HETATM':
                write_all += line
    with open(npath, 'w') as np:
        np.write(write_all)
    return 1


def main(list_path, pdb_path, new_pdb_path):
    '''
    remove HETATM in PDB
    '''
    if not os.path.exists(new_pdb_path):
        os.mkdir(new_pdb_path)
    with open(list_path, 'r') as li:
        write_all = ''
        for line in li:
            name = line[:4].upper()
            # print('deal with {}'.format(name))
            flag = remove_HETATM(pdb_path + name + '.pdb', new_pdb_path + name + '.pdb')


if __name__ == '__main__':
    main('./data/724_data_list.txt', './data/pdb/', './data/pdb_update/')
    # remove_HETATM('./ploymer/complex_pdb/5ACO.pdb','./ploymer/complex_pdb_update/5ACO.pdb')
